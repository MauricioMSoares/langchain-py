from langchain_classic.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    Language,
)
from langchain_classic.document_loaders import OnlinePDFLoader


chunk_size = 2000
chunk_overlap = 200

character_splitter = CharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""],
)

loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
data = loader.load()

# Recursive Splitter
recursive_splitter.split_text(data[0].page_content)
data = loader.load_and_split(text_splitter=recursive_splitter)

# Token Splitter
token_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
page = data[1]

token_split_page = token_splitter.split_text(page.page_content)
token_split_page

data = loader.load_and_split(text_splitter=token_splitter)
print(data)

# Code Splitter
RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
PYTHON_CODE = """
def merge_sort(arr):
   # Base case: If the array has 1 or no elements, it is already sorted
   if len(arr) <= 1:
       return arr
   # Divide: Split the array into two halves
   mid = len(arr) // 2
   left_half = merge_sort(arr[:mid])
   right_half = merge_sort(arr[mid:])
   # Conquer: Merge the sorted halves
   return merge(left_half, right_half)
def merge(left, right):
   sorted_array = []
   i = j = 0
   # Merge elements from both halves in sorted order
   while i < len(left) and j < len(right):
       if left[i] <= right[j]:
           sorted_array.append(left[i])
           i += 1
       else:
           sorted_array.append(right[j])
           j += 1
   # Append remaining elements from either half
   sorted_array.extend(left[i:])
   sorted_array.extend(right[j:])
   return sorted_array
# Example usage
arr = [38, 27, 43, 3, 9, 82, 10]
print("Original Array:", arr)
sorted_arr = merge_sort(arr)
print("Sorted Array:", sorted_arr)
"""

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=500, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
python_docs
