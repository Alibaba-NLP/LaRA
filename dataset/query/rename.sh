#!/bin/bash

for file in *; do
  if [[ -f "$file" ]]; then
    # 替换文件名中的 "novelette" 和 "novel"
    new_name=$(echo "$file" | sed 's/_fake//g')

    # 仅在新旧文件名不相等时重命名文件
    if [[ "$new_name" != "$file" ]]; then
      mv "$file" "$new_name"
    fi
  fi
done