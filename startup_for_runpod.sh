#!/usr/bin/env bash
set -euo pipefail

echo "[startup] begin"

# Force IPv4 (avoids some DC v6 quirks)
echo 'Acquire::ForceIPv4 "true";' >/etc/apt/apt.conf.d/99force-ipv4

APT_OPTS=(-o Acquire::Retries=5 -o Acquire::https::Timeout=25)

# Ensure security uses https
sed -i 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list || true

# Candidate HTTPS mirrors to try
MIRRORS=(
  "mirrors.edge.kernel.org/ubuntu"
  "mirror.math.princeton.edu/pub/ubuntu"
  "mirrors.rit.edu/ubuntu"
  "us.archive.ubuntu.com/ubuntu"
  "archive.ubuntu.com/ubuntu"            # last resort (may be http->https swapped)
)

# Replace any existing archive entry with first reachable mirror
REPLACED=0
for M in "${MIRRORS[@]}"; do
  sed -i 's|http://archive.ubuntu.com/ubuntu|https://'"$M"'|g' /etc/apt/sources.list
  sed -i 's|https://archive.ubuntu.com/ubuntu|https://'"$M"'|g' /etc/apt/sources.list
  sed -i 's|https://azure.archive.ubuntu.com/ubuntu|https://'"$M"'|g' /etc/apt/sources.list
  echo "[startup] trying mirror https://$M"
  if apt-get "${APT_OPTS[@]}" update -y; then
    echo "[startup] using $M"
    REPLACED=1
    break
  fi
done

if [ "$REPLACED" -eq 0 ]; then
  echo "[startup] WARNING: all mirrors failed; continuing with cached indexes if any"
fi

# Base packages (install individually to avoid full fail)
PKGS=(tree ffmpeg git-lfs libsndfile1 espeak-ng curl)
for p in "${PKGS[@]}"; do
  if ! DEBIAN_FRONTEND=noninteractive apt-get "${APT_OPTS[@]}" install -y --no-install-recommends "$p"; then
    echo "[startup] skip $p (install failed)"
  fi
done

command -v git-lfs >/dev/null 2>&1 && git lfs install || true

# SSH keys
mkdir -p /root/.ssh
if [ -f /workspace/ssh_backup/id_ed25519 ]; then
  cp /workspace/ssh_backup/id_ed25519 /root/.ssh/id_ed25519
  cp /workspace/ssh_backup/id_ed25519.pub /root/.ssh/id_ed25519.pub
  chmod 600 /root/.ssh/id_ed25519
  chmod 644 /root/.ssh/id_ed25519.pub
  touch /root/.ssh/known_hosts
  ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null || true
  echo "[startup] ssh key installed"
else
  echo "[startup] /workspace/ssh_backup/id_ed25519 not found (skipping SSH setup)"
fi

echo "[startup] done"
