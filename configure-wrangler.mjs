
import { readFileSync, writeFileSync } from "node:fs";

const path = ".output/server/wrangler.json";
const config = JSON.parse(readFileSync(path, "utf8"));

config.observability = {
  logs: {
    enabled: true,
    invocation_logs: true,
    persist: true,
  },
};

writeFileSync(path, JSON.stringify(config, null, 2) + "\n");
console.log("Cloudflare observability enabled.");
