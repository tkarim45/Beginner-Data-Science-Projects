# Miscellaneous Applied

Diverse applied techniques: association mining, A/B testing, graph analytics, plagiarism detection, QR codes, web scraping, PDF parsing, and vision-language captioning.

[Back to all projects](../README.md)

| Project | Description |
|---|---|
| [Market Basket Analysis](Market%20Basket%20Analysis) | Apriori association rules on UCI Online Retail; top rule lift ~15 (Regency teacup set). |
| [A-B Test Analysis](A-B%20Test%20Analysis) | Seeded experiment, two-proportion z-test / chi-square / CI / power; recovers the true +0.018 effect. |
| [Social Network Analysis](Social%20Network%20Analysis) | SNAP Facebook (4,039 nodes / 88k edges); clustering 0.61, 13 communities, node 107 top betweenness. |
| [Plagiarism Detector](Plagiarism%20Detector) | TF-IDF cosine + char n-gram Jaccard on MRPC; best F1 0.82 at cosine >= 0.30. |
| [QR Code Generator and Reader](QR%20Code%20Generator%20and%20Reader) | qrcode + OpenCV; 8/8 payload round-trip, robust to noise via Reed-Solomon EC. |
| [Supply Chain Demand Prediction](Supply%20Chain%20Demand%20Prediction) | DataCo daily demand; Ridge MAE 36.4 beats naive 42.0 (~13%) on a near-constant series. |
| [Web Scraping Job Postings](Web%20Scraping%20Job%20Postings) | Live RemoteOK API scrape (stdlib urllib) + tag/company analysis. |
| [PDF Table Extractor](PDF%20Table%20Extractor) | pdfplumber extracts tables from a generated financial report PDF into DataFrames. |
| [Personal Finance Tracker](Personal%20Finance%20Tracker) | Seeded simulated transactions; category classifier 0.31 (amount is a weak feature). |
| [Image Captioning](Image%20Captioning) | Pretrained ViT-GPT2 transfer learning; fluent zero-training captions plus an honest miss. |

_10 projects in this category._
