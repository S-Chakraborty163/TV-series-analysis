import gradio as gr
from theme_classifier import ThemeClassifier
from character_network import CharacterNetworkGenerator, NamedEntityRecognizer
from text_classification import JutsuClassifier
from dotenv import load_dotenv
load_dotenv()
import os
from character_chatbot import CharacterChatbot

def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = theme_list_str.split(",")
    # Strip whitespace from themes
    theme_list = [theme.strip() for theme in theme_list]
    
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    themes_to_plot = [theme for theme in theme_list if theme.lower() != 'dialogue']
    
    available_themes = [theme for theme in themes_to_plot if theme in output_df.columns]
    
    if not available_themes:
        output_df_for_plot = output_df[[col for col in output_df.columns if col != 'dialogue']]
    else:
        output_df_for_plot = output_df[available_themes]

    output_df_for_plot = output_df_for_plot.sum().reset_index()
    output_df_for_plot.columns = ["Theme", "Score"]

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(output_df_for_plot["Theme"], output_df_for_plot["Score"])
    plt.title("Series Themes", fontsize=16)
    plt.xlabel("Theme", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    return plt.gcf()

def get_character_network(subtitles_path, ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html

def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
     jutsu_classifier = JutsuClassifier(model_path=text_classification_model, data_path=text_classification_data_path, huggingface_token=os.getenv("huggingface_token"))

     output = jutsu_classifier.classify_jutsu(text_to_classify)
     output = output[0]
     return output

def chat_with_character_chatbot(message, history):
    character_chatbot = CharacterChatbot("Soumyajit900000/Naruto-Llama-3-8b", huggingface_token=os.getenv("huggingface_token"))

    output = character_chatbot.chat(message, history)
    output = output['content'].strip()
    return output


def main():
    with gr.Blocks() as iface:
        with gr.Row():
           with gr.Column():
                gr.HTML("<h1>Theme Classification (zero shot Classifiers)</h1>")
                with gr.Row():
                        with gr.Column():
                            plot = gr.Plot()
                        with gr.Column():
                            theme_list = gr.Textbox(label="Themes")
                            subtitles_path = gr.Textbox(label="Subtitles or script Path")
                            save_path = gr.Textbox(label="Save Path")
                            get_themes_button = gr.Button("Get Themes")
                            get_themes_button.click(
                                 get_themes,
                                 inputs=[theme_list,subtitles_path, save_path], outputs=[plot]
                            )

        # character network sec:
        
        with gr.Row():
           with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                        with gr.Column():
                            network_html = gr.HTML()
                        with gr.Column():
                            subtitles_path = gr.Textbox(label="Subtitles or script Path")
                            ner_path = gr.Textbox(label="NERs save path")
                            get_network_graph_button = gr.Button("Get Character Network")
                            get_network_graph_button.click(get_character_network,inputs=[subtitles_path, ner_path], outputs=[network_html])
        
        # Text Classification with LLMs sec:
        
        with gr.Row():
           with gr.Column():
                gr.HTML("<h1>Text Classification with LLMs</h1>")
                with gr.Row():
                        with gr.Column():
                            text_classification_output = gr.Textbox(label="Text Classification Output")
                        with gr.Column():
                            text_classification_model = gr.Textbox(label="Model Path")
                            text_classification_data_path = gr.Textbox(label="Data path")
                            text_to_classify = gr.Textbox(label="Text Input")
                            classify_text_button = gr.Button("Classify Text (jutsu)")
                            classify_text_button.click(classify_text ,inputs=[text_classification_model, text_classification_data_path, text_to_classify], outputs=[text_classification_output])


        # Character Chatbot sec:

        with gr.Row():
           with gr.Column():
                gr.HTML("<h1>Character Chatbot</h1>")
                gr.ChatInterface(chat_with_character_chatbot)
                

    iface.launch(share=True)


if __name__ == "__main__":
    main()

