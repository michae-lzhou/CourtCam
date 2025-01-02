for file in *.txt; do
    if grep -q '^0' "$file"; then
        grep '^0' "$file" > temp && mv temp "$file"
    else
        > "$file"
    fi
done
