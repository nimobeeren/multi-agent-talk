import {
  Agent,
  run,
  setDefaultOpenAIClient,
  setOpenAIAPI,
  setTracingDisabled,
  tool,
} from "@openai/agents";
import "dotenv/config";
import { AzureOpenAI } from "openai";
import { z } from "zod";

const openai = new AzureOpenAI();
setDefaultOpenAIClient(openai);
setOpenAIAPI("chat_completions");
setTracingDisabled(true);

const agent = new Agent({
  name: "Single Agent",
  instructions: "You are a helpful assistant",
  tools: [
    tool({
      name: "addNumbers",
      description: "add two numbers together",
      parameters: z.object({ a: z.number(), b: z.number() }),
      execute: async ({ a, b }) => a + b,
    }),
  ],
});

const result = await run(agent, "What's 2+2?");
console.log(result.finalOutput);
