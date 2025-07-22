import "dotenv/config";
import { Agent, run, setDefaultOpenAIClient, setOpenAIAPI, setTracingDisabled } from "@openai/agents";
import { AzureOpenAI } from "openai";

const openai = new AzureOpenAI();
setDefaultOpenAIClient(openai);
setOpenAIAPI("chat_completions");
setTracingDisabled(true);

const agent = new Agent({
  name: "Single Agent",
  instructions: "You are a helpful assistant"
});

const result = await run(agent, "Write a haiku about recursion in programming")
console.log(result.finalOutput);
