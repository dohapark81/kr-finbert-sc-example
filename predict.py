from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-FinBert-SC')
model = AutoModelForSequenceClassification.from_pretrained('snunlp/KR-FinBert-SC')

classifier = pipeline(task='text-classification', model=model, tokenizer=tokenizer)

inputs = [
    "“이걸로 아이폰 이길 수 있겠어?” 삼성 ‘최고폰’ 유출…이렇게 생겼다",
    "삼성전자의 내년 상반기를 책임질 플래그십 스마트폰 ‘갤럭시S24 시리즈’의 공식 렌더링 이미지가 유출됐다.",
    "부산엑스포 역전가능~",
    "결과는 개 쪽팔림~",
    "역대급 불황에…삼성 반도체 성과급 ‘0원’, 스마트폰은 ‘방긋’",
    "삼성보다 먼저…애플 '450만원짜리 야심작' 승부수 던졌다",
    "호구들아 많이사라",
    "갤럭시 s24 AI 기능 미온적인 반응",
    "삼성전자 갤럭시S24 울트라 원가 부담 커진다, 노태문 가격 경쟁력 유지 총력",
    "역대급 불황에…삼성 반도체 성과급 ‘0원’, 스마트폰은 ‘방긋’",
    "삼성전자, 올해 마지막 날 7만8500원으로 마감…2년 만의 최고가"
]

outputs = classifier(inputs)

for i in range(len(inputs)):
    print(inputs[i], outputs[i])
