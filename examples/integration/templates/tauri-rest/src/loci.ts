import { invoke } from "@tauri-apps/api/core";

export type LociInfo = {
  status: string;
  version: string;
  n_vocab: number;
  n_ctx_train: number;
  n_embd: number;
};

export async function lociHealth(): Promise<{ status: string }> {
  return invoke("loci_health");
}

export async function lociInfo(): Promise<LociInfo> {
  return invoke("loci_info");
}

export async function lociGenerate(
  prompt: string,
  maxTokens = 256,
  temperature = 0.7
): Promise<string> {
  return invoke("loci_generate", {
    prompt,
    maxTokens,
    temperature,
  });
}
