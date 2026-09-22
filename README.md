# LLM Next Token Probability Sampling

This example demonstrates the core mechanism of Large Language Models (LLMs) as described in the article: they don't "write" text but instead predict the probability distribution for the next token (word or word-piece). It simulates an LLM returning these probabilities and then shows two methods of selecting the next token: greedy sampling (always picking the most probable) and probabilistic sampling (randomly picking based on the probabilities). This illustrates how LLMs generate seemingly coherent text through a mathematical, probabilistic process.

## Language

`python`

## How to Run

Save the code as `main.py`.
Run from your terminal: `python main.py`

## Original Article

This example accompanies the Turkish article: [Büyük Dil Modelleri (LLM) Metin Yazmaz: Olasılıkları Nasıl Döndürür?](https://fatihsoysal.com/blog/buyuk-dil-modelleri-llm-metin-yazmaz-olasiliklari-nasil-dondurur/).

## License

MIT — see [LICENSE](LICENSE).
