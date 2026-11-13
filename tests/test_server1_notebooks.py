"""Validate notebook vector-store and preview queries without external AI calls."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from aperturedb.Connector import Connector
from aperturedb.LangChain import ApertureDB

ROOT = Path(__file__).resolve().parents[1] / 'notebooks/semantic_search'


class LocalEmbeddings(Embeddings):
    def embed_query(self, text):
        return [1.] + [0.] * 1023

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


@pytest.mark.parametrize('name', ['website_ingest', 'website_search',
                                  'cohere_wikipedia_ingest', 'cohere_wikipedia_search'])
def test_notebook_vectorstore_and_native_previews(name):
    if not os.getenv('APERTUREDB_TEST_PORT'):
        pytest.skip('Set APERTUREDB_TEST_PORT to a disposable native server 1.0')
    client = Connector(host='127.0.0.1', port=int(os.environ['APERTUREDB_TEST_PORT']),
                       user='admin', password='admin', use_ssl=False)
    assert client.hello_version == 2
    descriptor_class = 'ApplicationNotebook' + uuid4().hex
    store = None
    try:
        namespace = dict(embeddings=LocalEmbeddings(), embeddings_dim=1024,
                         DESCRIPTOR_CLASS=descriptor_class,
                         args=SimpleNamespace(descriptorset=descriptor_class))
        def factory(**kwargs):
            return ApertureDB(client=client, **kwargs)
        factory.list_vectorstores = lambda: ApertureDB.list_vectorstores(client=client)
        namespace['ApertureDB'] = factory
        notebook = json.loads((ROOT / (name + '.ipynb')).read_text())
        trees=[]
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code' or ''.join(cell['source']).lstrip().startswith(('%', '!')):
                continue
            tree=ast.parse(''.join(cell['source']))
            trees.append(tree)
        constructor = next(node for tree in trees for node in ast.walk(tree)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                           and node.func.id == 'ApertureDB')
        store = eval(compile(ast.Expression(constructor), name, 'eval'), namespace)
        store.add_documents([Document(page_content='rice', metadata={'title': 'Rice'}),
                             Document(page_content='soup', metadata={'title': 'Soup'})])
        namespace.update(vectorstore=store, client=client, offset=0, json=json)
        assert store.as_retriever(search_kwargs={'k': 1}).invoke('rice')
        for tree in trees:
            # Execute the actual notebook's raw metadata preview dictionary.
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'query' for t in node.targets):
                    if isinstance(node.value, ast.List):
                        query = eval(compile(ast.Expression(node.value), name, 'eval'), namespace)
                        response, _ = client.query(query)
                        assert response[0]['FindDescriptor']['status'] == 0, response
                        assert response[0]['FindDescriptor']['returned'] == 2
            code = ast.unparse(tree)
            if 'list_vectorstores' in code:
                exec(compile(tree, name, 'exec'), namespace)
    finally:
        if store is not None:
            ApertureDB.delete_vectorstore(descriptor_class, client=client)
        client.close()
