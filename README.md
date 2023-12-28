# snunlp/KR-FinBert-SC
neutral, positive, negative로 0.0 - 1.0까지 값으로 나오는 모델 (뉴스분석에 적합해 보임)
개발환경: python 3.10 기준 (아나콘다, 파이참 사용)

모델: snunlp/KR-FinBert-SC

```
conda install -y pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.36.2 chardet==5.2.0
```