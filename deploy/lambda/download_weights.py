"""Download both Decider repositories into the Hugging Face cache at pinned revisions.

Runs during the image build. It writes `refs/main` so that offline mode resolves "main" to the pinned
revision instead of whatever is on the Hub on build day.
"""

import os

from huggingface_hub import snapshot_download

PINNED_REVISIONS = {
    "StrandsAgents/strands-decider-2B-hobson-v19": "bb282d786bc251fd4e3068de3ada9ddbb38127cd",
    "Qwen/Qwen3.5-2B-Base": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
}


def main() -> None:
    for repo, sha in PINNED_REVISIONS.items():
        path = snapshot_download(repo, revision=sha)
        refs_dir = os.path.join(path, "..", "..", "refs")
        os.makedirs(refs_dir, exist_ok=True)
        with open(os.path.join(refs_dir, "main"), "w") as handle:
            handle.write(sha)
        print(repo, "->", path)


if __name__ == "__main__":
    main()
