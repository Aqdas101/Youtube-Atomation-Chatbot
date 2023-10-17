class WriteVideoDescription:
    def __init__(self, openai_api_key):
        from langchain.llms import OpenAI
        from langchain.chains.summarize import load_summarize_chain
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.globals import set_llm_cache
        from langchain.cache import InMemoryCache
        from langchain.prompts import PromptTemplate

        set_llm_cache(InMemoryCache())
        
        llm = OpenAI(model_name = 'text-davinci-003', openai_api_key=openai_api_key)
        self.chain = load_summarize_chain(llm, chain_type="map_reduce")
        template = '''Write a short and informative YouTube Video Description. Write in 2nd-peron perspective. write what this video about.

        "{text}"
        
        SUMMARY:
        '''
        prompt = PromptTemplate(
            input_variables = ['text'],
            template = template
        )
        self.chain.llm_chain.prompt = prompt
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 20 )



    def parse(self, URL):
        
        from langchain.document_loaders import YoutubeLoader
        loader = YoutubeLoader.from_youtube_url(
            URL, add_video_info=True
        )
        data = loader.load()
        split_doc = self.text_splitter.split_documents(data)
        
        return self.chain.run(split_doc)