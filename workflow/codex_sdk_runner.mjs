#!/usr/bin/env node

import fs from "node:fs/promises";
import process from "node:process";

function parseArgs(argv) {
  const options = {
    cwd: process.cwd(),
    model: "gpt-5.4",
    reasoningEffort: "medium",
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--cwd") {
      options.cwd = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === "--prompt-file") {
      options.promptFile = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === "--model") {
      options.model = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === "--reasoning-effort") {
      options.reasoningEffort = argv[index + 1];
      index += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (!options.promptFile) {
    throw new Error("Missing required --prompt-file argument.");
  }

  return options;
}

function extractText(value) {
  if (typeof value === "string") {
    return value;
  }
  if (!value || typeof value !== "object") {
    return "";
  }

  const directKeys = ["outputText", "output_text", "text", "content"];
  for (const key of directKeys) {
    if (typeof value[key] === "string" && value[key].trim()) {
      return value[key];
    }
  }

  if (Array.isArray(value.output)) {
    const text = value.output
      .map((item) => {
        if (!item || typeof item !== "object") {
          return "";
        }
        if (typeof item.text === "string") {
          return item.text;
        }
        if (Array.isArray(item.content)) {
          return item.content
            .map((part) => (typeof part?.text === "string" ? part.text : ""))
            .filter(Boolean)
            .join("\n");
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
    if (text.trim()) {
      return text;
    }
  }

  if (Array.isArray(value.messages)) {
    const text = value.messages
      .map((message) => {
        if (!message || typeof message !== "object") {
          return "";
        }
        if (typeof message.text === "string") {
          return message.text;
        }
        if (Array.isArray(message.content)) {
          return message.content
            .map((part) => (typeof part?.text === "string" ? part.text : ""))
            .filter(Boolean)
            .join("\n");
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
    if (text.trim()) {
      return text;
    }
  }

  return JSON.stringify(value, null, 2);
}

async function loadSdk() {
  try {
    const module = await import("@openai/codex-sdk");
    return module.Codex || module.default?.Codex || module.default;
  } catch (error) {
    throw new Error(
      "Unable to import @openai/codex-sdk. Install it in this repository before running workflow/run.py."
        + ` Original error: ${error.message}`
    );
  }
}

async function startThread(Codex, options) {
  const constructorCandidates = [
    {
      cwd: options.cwd,
      model: options.model,
      reasoningEffort: options.reasoningEffort,
      approvalPolicy: "never",
      sandboxMode: "workspace-write",
      config: {
        model: options.model,
        model_reasoning_effort: options.reasoningEffort,
        modelReasoningEffort: options.reasoningEffort,
        reasoningEffort: options.reasoningEffort,
      },
    },
    {
      cwd: options.cwd,
      config: {
        model: options.model,
        model_reasoning_effort: options.reasoningEffort,
      },
    },
    {},
  ];

  const threadCandidates = [
    {
      cwd: options.cwd,
      model: options.model,
      reasoningEffort: options.reasoningEffort,
      approvalPolicy: "never",
      sandboxMode: "workspace-write",
      config: {
        model: options.model,
        model_reasoning_effort: options.reasoningEffort,
      },
    },
    { cwd: options.cwd },
    undefined,
  ];

  let lastError = null;

  for (const constructorOptions of constructorCandidates) {
    try {
      const codex = new Codex(constructorOptions);
      if (typeof codex.startThread === "function") {
        for (const threadOptions of threadCandidates) {
          try {
            return threadOptions === undefined
              ? await codex.startThread()
              : await codex.startThread(threadOptions);
          } catch (error) {
            lastError = error;
          }
        }
      }
      if (typeof codex.createThread === "function") {
        for (const threadOptions of threadCandidates) {
          try {
            return threadOptions === undefined
              ? await codex.createThread()
              : await codex.createThread(threadOptions);
          } catch (error) {
            lastError = error;
          }
        }
      }
      lastError = new Error("Codex SDK object does not expose startThread() or createThread().");
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error("Unable to start a Codex SDK thread.");
}

async function runPrompt(thread, prompt) {
  if (typeof thread.run === "function") {
    return await thread.run(prompt);
  }
  if (typeof thread.send === "function") {
    return await thread.send(prompt);
  }
  throw new Error("Codex thread object does not expose run() or send().");
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const prompt = await fs.readFile(options.promptFile, "utf8");

  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not set.");
  }

  const Codex = await loadSdk();
  const thread = await startThread(Codex, options);
  const result = await runPrompt(thread, prompt);
  const text = extractText(result).trim();

  process.stdout.write(
    JSON.stringify(
      {
        text,
        model: options.model,
        reasoning_effort: options.reasoningEffort,
      },
      null,
      2
    ) + "\n"
  );
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exit(1);
});
