"""Rebuild paper/generated_numbers.tex from results/*.json."""

from pcm.report import write_generated_numbers

if __name__ == "__main__":
    print("wrote", write_generated_numbers())
