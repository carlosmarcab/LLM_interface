import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import json
import threading
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader

with open('config.json') as file:
    data = json.load(file)

openai_api_key = data['credentials']['OPENAI_API_KEY']
persist_directory = "db"

FACTUAL_PROMPT = "Answer the user questions truthfully. " \
                 "Answer exclusively to what you have been requested." \
                 "Only use the information in the CONTEXT to elaborate your answer or answer 'I dont know'. " \
                 "Quote verbatim the text relevant to the user's query. " \
                 "Quote only the necessary text." \
                 "Use the minimum amount of text necessary to answer the user's query" \
                 "Use ellipsis '(...)' to ommit unnecesary parts of a quote" \
                 "Minimize non quote text" \
                 "If you don't know the answer to something, say 'I dont know'."
CREATIVE_PROMPT = "Answer the user questions truthfully and complete the task." \
                  "Use the information in the CONTEXT as much as possible to elaborate your answer." \
                  "You may reason your answer and summarize the CONTEXT if needed" \
                  "You may complete information or add context if necessary" \
                  "You may quote non-literally and connect ideas from the context if needed to complete the task"


class LLModel:
    """LLM wrapper class to abstract completions"""

    def __init__(self, model: str, mode="general"):
        if model == "gpt-3.5-turbo" or model == "gpt-4" or model == "gpt-3.5-turbo-16k":
            self.model = model
            self.system_message = []
        else:
            raise ValueError("Model type is not valid")
        if mode == "general":
            self.llm = ChatOpenAI(model=self.model, temperature=0.4, openai_api_key=openai_api_key)
        elif mode == "factual":
            self.system_message = [SystemMessage(content=FACTUAL_PROMPT)]
            self.llm = ChatOpenAI(model=self.model, temperature=0, openai_api_key=openai_api_key)
        else:
            self.system_message = [SystemMessage(content=CREATIVE_PROMPT)]
            self.llm = ChatOpenAI(model=self.model, temperature=0.3, openai_api_key=openai_api_key)

    def complete(self, conversation):
        if self.system_message:
            conversation = self.system_message + conversation
        return self.llm(conversation).content


class GPTInterface:
    """The main class for a graphical user interface for interacting with GPT models."""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("GPT interface")
        self.window.geometry("800x600")
        self.window.iconbitmap("openai.ico")

        self.model = tk.StringVar(value="gpt-3.5-turbo")
        self.mode = None
        self.database = None
        self.conversation = []
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.file_path = None
        self.loader = None

        self.create_widgets()

        self.window.mainloop()

    def create_widgets(self):
        """Create and configure the widgets for the GUI."""

        # Configuring styles
        style = ttk.Style()
        style.theme_use('winnative')
        style.configure("TFrame", background="#c5d6e9")
        style.configure("Fileframe.TFrame", background="#a8bbd4")
        style.configure("TButton", background="#d4ddea", foreground="#000")
        style.configure("TRadiobutton", background="#c5d6e9")
        style.configure("TEntry", background="#d9e5f9")
        style.configure("TLabel", background="#c5d6e9")
        style.configure("blue.Horizontal.TProgressbar", troughcolor="#d4ddea", background="#8696aa",
                        foreground="#8696aa")

        # Creating and packing widgets
        self.file_frame = ttk.Frame(self.window, padding=10, style="Fileframe.TFrame")
        self.file_frame.pack(fill="x")

        self.upload_button = ttk.Button(self.file_frame, text="Upload File", command=self.upload_file)
        self.upload_button.pack(side="left", padx=10)

        self.file_label = ttk.Label(self.file_frame, text="Uploaded File: None", background="#a8bbd4",
                                    foreground="#000")
        self.file_label.pack(side="left", padx=10)

        self.progress_bar = ttk.Progressbar(self.file_frame, length=200, mode="indeterminate", orient='horizontal',
                                            style="blue.Horizontal.TProgressbar")
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = 100

        self.process_button = ttk.Button(self.file_frame, text="Process File", command=self.process_file)
        self.process_button.pack(side="left", padx=10)

        self.input_output_frame = ttk.Frame(self.window, padding=10)
        self.input_output_frame.pack(fill="both", expand=True)

        self.input_label = ttk.Label(self.input_output_frame, text="User Input:")
        self.input_label.grid(row=0, column=0, sticky='w')

        self.input_entry = tk.Text(self.input_output_frame, width=40, height=3, font=('Calibri', 12))
        self.input_entry.grid(row=1, column=0, sticky='we')

        self.send_button = ttk.Button(self.input_output_frame, text="Send Input", command=self.send_input)
        self.send_button.grid(row=2, column=0, sticky='e')

        self.radio_frame = ttk.Frame(self.input_output_frame, padding=10)
        self.radio_frame.grid(row=2, column=0, sticky='w')

        self.gpt_35_button = ttk.Radiobutton(self.radio_frame, text="GPT-3.5", variable=self.model,
                                             value="gpt-3.5-turbo")
        self.gpt_35_button.pack(side="left", padx=10)
        self.gpt_4_button = ttk.Radiobutton(self.radio_frame, text="GPT-4", variable=self.model, value="gpt-4")
        self.gpt_4_button.pack(side="left", padx=5)

        self.output_label = ttk.Label(self.input_output_frame, text="Program Output:")
        self.output_label.grid(row=4, column=0, sticky='w')

        self.output_text = tk.Text(self.input_output_frame, height=5, width=40, font=('Calibri', 12), wrap="word")
        self.output_text.grid(row=5, column=0, sticky='nsew')

        self.output_scroll = tk.Scrollbar(self.input_output_frame, command=self.output_text.yview)
        self.output_scroll.grid(row=5, column=1, sticky='ns')

        self.output_text.configure(yscrollcommand=self.output_scroll.set)
        self.output_text.configure(state='disabled')
        self.output_text.tag_configure('user', background='#f0f5fA')  # very clear blue
        self.output_text.tag_configure('gpt', background='#e3e9f1', justify='right')  # slightly darker blue

        self.input_output_frame.grid_columnconfigure(0, weight=1)
        self.input_output_frame.grid_rowconfigure(5, weight=1)

        self.clear_button = ttk.Button(self.input_output_frame, text="Clear Output", command=self.clear_output)
        self.clear_button.grid(row=6, column=0, pady=10)  # add pady for padding
        self.input_output_frame.grid_columnconfigure(0, weight=1)  # add weight to the column to center the button

        # Creating the main menu
        self.create_main_menu()

    def create_main_menu(self):
        """Create the main menu for the application."""

        menu = tk.Menu(self.window)
        self.window.config(menu=menu)

        filemenu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Menu", menu=filemenu)
        filemenu.add_command(label="Open Database", command=self.open_database)
        filemenu.add_command(label="Clear Database", command=self.clear_database)
        filemenu.add_command(label="Factual Mode", command=lambda: self.set_mode("factual"))
        filemenu.add_command(label="Creative Mode", command=lambda: self.set_mode("creative"))

    def upload_file(self):
        """Allow the user to select a file to upload."""

        try:
            file_path = filedialog.askopenfilename(title='Select a PDF File', filetypes=[('PDF Files', '*.pdf')])
            self.file_label.config(text="Uploaded File: " + file_path.split("/")[-1])
            self.loader = UnstructuredPDFLoader(file_path)
        except Exception as e:
            print(f"Error in uploading file: {e}")

    def process_file(self):
        """Start a new thread to process the file."""

        self.progress_bar.pack(side="left", padx=10)
        self.progress_bar.start(10)

        # Start a new thread to process the file
        thread = threading.Thread(target=self.process_file_thread)
        thread.start()

    def process_file_thread(self):
        """Process the file by loading, chunking, embedding and storing"""

        try:
            documents = self.loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            if not self.database:
                self.database = Chroma.from_documents(texts, self.embeddings, persist_directory=persist_directory)
            else:
                self.database.add_documents(texts)
            self.database.persist()
        except Exception as e:
            print(f"Error processing the file: {e}")
        finally:
            # Stop the progress bar and enable the button in the main thread
            self.window.after(0, self.stop_progress_bar)
            self.window.after(0, self.enable_process_button)

    def enable_process_button(self):
        """Enable the process file button."""

        self.process_button["state"] = "normal"

    def stop_progress_bar(self):
        """Stop and hide the progress bar."""

        self.progress_bar.stop()
        self.progress_bar.pack_forget()

    def send_input(self):
        """Function to send the input to the API and display it"""

        self.progress_bar.pack(side="left", padx=10)
        self.progress_bar.start(10)
        self.send_button["state"] = "disabled"

        # Take the user input and display it
        user_input = self.input_entry.get('1.0', 'end-1c')
        self.output_text.configure(state='normal')
        self.output_text.insert('end', 'USER:' + '\n', 'user')
        self.output_text.insert('end', user_input + '\n\n', 'user')
        self.output_text.configure(state='disabled')

        # Start a new thread to send the input
        thread = threading.Thread(target=self.send_input_thread)
        thread.start()

    def send_input_thread(self):
        """Send the user's input and display the output in a new thread."""

        user_input = self.input_entry.get('1.0', 'end-1c')
        self.input_entry.delete("1.0", "end")

        if not self.database:  # If no database is selected, create a simple conversation
            model = LLModel(self.model.get())
            self.conversation.append(HumanMessage(content=user_input))
            try:
                output = model.complete(self.conversation)
                agent = "GPT:"
                self.conversation.append(AIMessage(content=output))
            except Exception as e:
                agent = "SYSTEM:"
                output = "There has been an error contacting the LLM. Please retry."
                print(f"Error contacting the LLM: {e}")
        else:
            try:
                agent = "GPT:"
                model_to_use = self.model.get()
                if model_to_use == "gpt-3.5-turbo":
                    model_to_use == model_to_use + "-16k"
                    chunks = self.database.similarity_search(user_input, k=6)
                else:
                    chunks = self.database.similarity_search(user_input, k=4)
                prompt = "TASK: " + user_input + '\nCONTEXT: """\n'
                for doc in chunks:
                    prompt += doc.page_content + "\n\n---\n\n"
                prompt += '\n"""'
                model = LLModel(model_to_use, mode=self.mode)
                self.conversation.append(HumanMessage(content=prompt))
                output = model.complete(self.conversation)
                self.conversation.append(AIMessage(content=output))
            except Exception as e:
                agent = "SYSTEM:"
                output = "There has been an error contacting the LLM. Please retry."
                print(f"Error contacting the LLM: {e}")
                raise e

        self.output_text.configure(state='normal')
        self.output_text.insert('end', agent + '\n', 'gpt')
        self.output_text.insert('end', output + '\n\n', 'gpt')
        self.output_text.configure(state='disabled')

        # Stop the progress bar and enable the send button in the main thread
        self.window.after(0, self.stop_progress_bar)
        self.window.after(0, self.enable_send_button)

    def enable_send_button(self):
        """Enable the send button."""

        self.send_button["state"] = "normal"

    def open_database(self):
        """Allow the user to select a database directory."""

        try:
            db_path = filedialog.askdirectory()
            self.clear_output()
            self.database = Chroma(embedding_function=self.embeddings, persist_directory=db_path)
            print("Database imported successfully")
            self.mode = "factual"
        except Exception as e:
            print(f"Error in opening database: {e}")

    def set_mode(self, mode):
        self.mode = mode

    def clear_output(self):
        self.output_text.configure(state='normal')
        self.output_text.delete("1.0", "end")
        self.output_text.configure(state='disabled')
        self.conversation = []

    def clear_database(self):
        self.database = None
        self.clear_output()


if __name__ == "__main__":
    GPTInterface()
