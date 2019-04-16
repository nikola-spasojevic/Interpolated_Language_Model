import pickle

class TrieNode:
    def __init__(self):
        self.end = False
        self.children = {}

    def all_words(self, prefix):
        if self.end:
            yield prefix
        for letter, child in self.children.items():
            yield from child.all_words(prefix + letter)

class Trie:
    def __init__(self, vocabulary):
        self.root = TrieNode()
        for word in vocabulary:
            self.insert(word)

    def insert(self, word):
        curr = self.root
        for letter in word:
            node = curr.children.get(letter)
            if not node:
                node = TrieNode()
                curr.children[letter] = node
            curr = node
        curr.end = True

    def search(self, word):
        curr = self.root

        for letter in word:
            node = curr.children.get(letter)
            if not node:
                return False
            curr = node
        return curr.end

    def all_words_beginning_with_prefix(self, prefix):
        cur = self.root
        for c in prefix:
            cur = cur.children.get(c)
            if cur is None:
                return  # No words with given prefix

        yield from cur.all_words(prefix)

def main():
    with open('../bin/vocabulary.pkl', 'rb') as pickle_in:
            vocabulary = pickle.load(pickle_in, encoding='utf8')
    trie = Trie(vocabulary)
    with open('../bin/trie.pkl', 'wb') as output:
        pickle.dump(trie, output)
        output.close()

if __name__ == "__main__":
    main()