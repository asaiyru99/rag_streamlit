from functions import delete_all, delete_and_save, merge_pdf, convert_to_pdf, get_config

config = get_config()

from retrieval_gen import TextProcess, EmbeddedRAG
import streamlit as st
import time

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = None
if "confirmed" not in st.session_state:
    st.session_state["confirmed"] = False
if "chat_active" not in st.session_state:
    st.session_state["chat_active"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def response_generator(path, query):
    preprocessing = TextProcess(path, 10)
    data = preprocessing.chunk_dict_maker()
    middle_pipeline = EmbeddedRAG(data, query)
    answer = middle_pipeline.ask()
    response = answer
    #for word in response.split():
    yield response

# Define Main class
class Main:
    def __init__(self):
        self.upload_files()

        if st.session_state["uploaded_files"] and not st.session_state["confirmed"]:
            self.confirm_files()

        if st.session_state["confirmed"]:
            self.process_files()

        if st.session_state["chat_active"]:
            self.chat_interface()

        self.reset_app()

    def upload_files(self):
        st.subheader("Welcome to miniRAG!")
        uploaded_files = st.file_uploader(
            "Upload multiple documents", type=["txt", "pdf", "docx"], accept_multiple_files=True
        )
        if uploaded_files:
            st.session_state["uploaded_files"] = uploaded_files
            st.session_state["confirmed"] = False
            st.markdown("Files uploaded successfully!")

    def confirm_files(self):
        confirm = st.radio("Are the files are ready for processing?", ["No", "Yes"])
        if confirm == "Yes":
            st.session_state["confirmed"] = True
            st.markdown("Files are ready to be processed!")

    def process_files(self):
        if st.button("Yes, process the files"):
            time.sleep(1) 

            # Retrieve uploaded files from session state
            uploaded_files = st.session_state["uploaded_files"]


            st.write(f"Processing files")
            delete_and_save(uploaded_files, config['upload_path'])
            delete_all(config['processed_path'])
            convert_to_pdf(config['upload_path'], config['processed_path'])
            merge_pdf(config["processed_path"], config["merge_pdf"])

            st.markdown("I have processed the filessuccessfully! You may ask me a question")
            st.session_state["chat_active"] = True  # Activate chat after processing

    def chat_interface(self):

        if prompt := st.chat_input("What is your question?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = f"Processing your query: {prompt}"
            with st.chat_message("assistant"):
                st.markdown(response)
            question = prompt

            with st.chat_message("assistant"):
                response = st.write(response_generator(config['merge_pdf'], question))

            st.session_state.messages.append({"role": "assistant", "content": response})

    def reset_app(self):
        if st.button("Reset to upload new files"):
            st.session_state["uploaded_files"] = None
            st.session_state["confirmed"] = False
            st.session_state["chat_active"] = False
            st.session_state["messages"] = []
            st.components.v1.html("""<script type = "text/javascript">window.location.reload();</script>""", height = 0)  
            st.markdown("Please upload your files again.")


if __name__ == "__main__":
    Main()
