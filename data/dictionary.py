from typing import List, Sequence

class Dictionary:
    """Dictionary class for text recognition.
    
    Args:
        dict_file (str): Character dict file path where each line contains one character.
        with_start (bool): Whether to include start token. Defaults to False.
        with_end (bool): Whether to include end token. Defaults to False.
        same_start_end (bool): Whether start and end tokens are same. Defaults to False.
        with_padding (bool): Whether to include padding token. Defaults to False.
        with_unknown (bool): Whether to include unknown token. Defaults to False.
        start_token (str): Start token string. Defaults to '<BOS>'.
        end_token (str): End token string. Defaults to '<EOS>'.
        start_end_token (str): Combined start/end token string. Defaults to '<BOS/EOS>'.
        padding_token (str): Padding token string. Defaults to '<PAD>'.
        unknown_token (str): Unknown token string. Defaults to '<UKN>'.
    """

    def __init__(self,
                 dict_file: str,
                 with_start: bool = False,
                 with_end: bool = False,
                 same_start_end: bool = False,
                 with_padding: bool = True,
                 with_unknown: bool = True,
                 start_token: str = '<BOS>',
                 end_token: str = '<EOS>',
                 start_end_token: str = '<BOS/EOS>',
                 padding_token: str = '<PAD>',
                 unknown_token: str = '<UKN>') -> None:
        
        self.with_start = with_start
        self.with_end = with_end
        self.same_start_end = same_start_end
        self.with_padding = with_padding
        self.with_unknown = with_unknown
        self.start_end_token = start_end_token
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.unknown_token = unknown_token

        # Load dictionary from file
        self._dict = []
        with open(dict_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip('\r\n')
                if len(line) > 1:
                    raise ValueError(f'Each line should have 0 or 1 character, '
                                  f'got {len(line)} characters at line {line_num + 1}')
                if line != '':
                    self._dict.append(line)

        # Initialize char to index mapping
        self._char2idx = {char: idx for idx, char in enumerate(self._dict)}
        
        # Update dictionary with special tokens
        self._update_dict()
        
        # Check for duplicates
        assert len(set(self._dict)) == len(self._dict), \
            'Invalid dictionary: Contains duplicated characters.'

    @property
    def num_classes(self) -> int:
        """Number of classes including special tokens."""
        return len(self._dict)

    @property
    def dict(self) -> list:
        """Dictionary list including special tokens."""
        return self._dict

    def char2idx(self, char: str, strict: bool = True) -> int:
        """Convert character to index.

        Args:
            char (str): Character to convert.
            strict (bool): Whether to raise exception for unknown chars.

        Returns:
            int: Character index.
        """
        char_idx = self._char2idx.get(char, None)
        if char_idx is None:
            if self.with_unknown:
                return self.unknown_idx
            elif not strict:
                return None
            else:
                raise Exception(f'Character: {char} not in dictionary. '
                              'Please check labels and dictionary file, '
                              'or set "with_unknown=True"')
        return char_idx

    def str2idx(self, string: str) -> List:
        """Convert string to index list.

        Args:
            string (str): String to convert.

        Returns:
            list: List of character indices.
        """
        indices = []
        for char in string:
            char_idx = self.char2idx(char)
            if char_idx is None:
                if self.with_unknown:
                    continue
                raise Exception(f'Character: {char} not in dictionary. '
                              'Please check labels and dictionary file, '
                              'or set "with_unknown=True"')
            indices.append(char_idx)
        return indices

    def idx2str(self, indices: Sequence[int]) -> str:
        """Convert index list to string.

        Args:
            indices (list[int]): List of indices to convert.

        Returns:
            str: Converted string.
        """
        assert isinstance(indices, (list, tuple))
        string = ''
        for idx in indices:
            assert idx < len(self._dict), \
                f'Index {idx} out of range! Must be less than {len(self._dict)}'
            string += self._dict[idx]
        return string

    def _update_dict(self):
        """Update dictionary with special tokens."""
        # Add start/end tokens
        self.start_idx = None
        self.end_idx = None
        if self.with_start and self.with_end and self.same_start_end:
            self._dict.append(self.start_end_token)
            self.start_idx = len(self._dict) - 1
            self.end_idx = self.start_idx
        else:
            if self.with_start:
                self._dict.append(self.start_token)
                self.start_idx = len(self._dict) - 1
            if self.with_end:
                self._dict.append(self.end_token)
                self.end_idx = len(self._dict) - 1

        # Add padding token
        self.padding_idx = None
        if self.with_padding:
            self._dict.append(self.padding_token)
            self.padding_idx = len(self._dict) - 1

        # Add unknown token
        self.unknown_idx = None
        if self.with_unknown and self.unknown_token is not None:
            self._dict.append(self.unknown_token)
            self.unknown_idx = len(self._dict) - 1

        # Update char to index mapping
        self._char2idx = {char: idx for idx, char in enumerate(self._dict)}