#!/bin/sh
# Installs the Expert Craft skill so an AI agent can use it:
#   curl -fsSL https://raw.githubusercontent.com/Tharunmaks/Stream-/claude/expertstream-moe-android-oegt59/scripts/install-skill.sh | sh
# Default target: ~/.claude/skills/expert-craft (Claude Code). Pass another skills folder as the first argument for other agents.
set -e
BRANCH="claude/expertstream-moe-android-oegt59"
DEST="${1:-$HOME/.claude/skills/expert-craft}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if command -v git >/dev/null 2>&1; then
  git clone -q --depth 1 --branch "$BRANCH" https://github.com/Tharunmaks/Stream-.git "$TMP/src"
else
  curl -fsSL "https://codeload.github.com/Tharunmaks/Stream-/tar.gz/refs/heads/$BRANCH" | tar -xz -C "$TMP"
  mv "$TMP"/Stream-* "$TMP/src"
fi
[ -f "$TMP/src/craft/skills/expert-craft/SKILL.md" ] || { echo "download failed: skill not found" >&2; exit 1; }
mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -R "$TMP/src/craft/skills/expert-craft" "$DEST"
echo "Expert Craft skill installed in $DEST"
echo "Now ask your AI: \"Use the expert-craft skill to plan and train a 70B model on this device.\""
command -v python3 >/dev/null 2>&1 || echo "Note: Python 3 is needed (Termux: pkg install -y python python-numpy clang)."
