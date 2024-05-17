import gradio as gr
from transformers import pipeline

# Sentiment analysis
def sentiment_analysis(input):
    cls_sentiment = pipeline("sentiment-analysis",
                             model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa")

    result = cls_sentiment(input)
    label = result[0]['label']
    score = result[0]['score']

    return f"Sentiment: {label} with confidence score: {score:.2f}"
    

# Named Entity Recognition
def ner(input):
    cls_ner = pipeline("ner",
                       model="AptaArkana/indonesian_nergrit_bert_base_multilingual_cased",
                       tokenizer="AptaArkana/indonesian_nergrit_bert_base_multilingual_cased",
                       aggregation_strategy="simple")

    entity_result = cls_ner(input)
    
    return {"text": input, "entities": entity_result}


# Language identification
def language_identification(input):
    cls_lang = pipeline("text-classification",
                        model="ivanlau/language-detection-fine-tuned-on-xlm-roberta-base")

    result = cls_lang(input)
    label = result[0]['label']
    score = result[0]['score']

    return f"Language: {label} with confidence score: {score:.2f}"

# Code-mixed language identification
def code_mixed_lid(input):
    cls_indojave_indobertweet = pipeline("token-classification",
                                         model="fathan/ijelid-ft-indojave-indobertweet",
                                         aggregation_strategy="simple")

    result_list = []
    result = cls_indojave_indobertweet(input)

    for item in result:
        tokens = item['word'].split()
        tag = item['entity_group']
        item_length = len(tokens)
        if item_length > 1:
            for token in tokens:
                result_list.append((token, tag))
        else:
            result_list.append((item['word'], tag))

    return result_list


# Define shared components and title
shared_title = "NLP Demo"
shared_textbox = gr.Textbox(placeholder="Enter sentence here...", lines=4, max_lines=5, label="Input Text")



with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Tab("Sentiment Analysis"):
        gr.Interface(
            fn=sentiment_analysis,
            inputs=shared_textbox,
            outputs="text",
            title="Sentiment Analysis for Indonesian",
            allow_flagging='never',
            examples=[
                "Keren banget hasilnya, aku suka!",
                "Aku benci sekali dengan hasil ini, sangat buruk!"
            ]
        )

    with gr.Tab("Named Entity Recognition"):
        gr.Interface(
            fn=ner,
            inputs=shared_textbox,
            outputs=gr.HighlightedText(),
            title="Named Entity Recognition for Indonesian",
            allow_flagging='never',
            examples=[
                "Jakarta adalah ibu kota Indonesia.",
                "Indonesia merdeka pada 17 Agustus 1945.",
                "Beli laptop MacBook Pro di Apple Store."
            ]
        )

    with gr.Tab("Language Identification"):
        gr.Interface(
            fn=language_identification,
            inputs=shared_textbox,
            outputs="text",
            title="Language Identification",
            allow_flagging='never',
            examples=[
                "Saya belajar Python secara mandiri.",
                "Buongiorno e grazie a tutti",
                "Thank you for joining the meeting."
            ]
        )

    with gr.Tab("Code-mixed Language Identification"):
        gr.Interface(
            fn=code_mixed_lid,
            inputs=shared_textbox,
            outputs=gr.HighlightedText(),
            title="Code-mixed LID for Indonesian, Javanese, and English",
            allow_flagging='never',
            examples=[
                "Guys, sorry iki aku belum bisa join online meetingnya",
                "Ayo jalan2 ke Singapore aja guys!",
                "Pancen wong iki, bales replynya lamaaa bgt!"
            ]
        )

# Launch the application
if __name__ == "__main__":
    demo.launch()
