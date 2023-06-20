from collections import OrderedDict
import datetime
import numpy as np
import re
import warnings

class ChbMetadata:
    """
    Class for handling the labels
    """
    def __init__(self, filename):
        self._filename = filename
        self._file = open(filename, 'r')
        self._parse_file(self._file)
        self._file.close()

    def _parse_file(self, file_obj):
        """
        Parse the summary seizure and non-seizures file
        
        Input:
            file_obj, opened summary file
        """
        # Split file into blocks
        data = file_obj.read()
        blocks = data.split('\n\n')

        # Block channels
        self._channel_names = self._parse_channel_names(blocks[1])

        # Block metadata
        self._metadata_store = self._parse_file_metadata(blocks[2:])


    def _parse_channel_names(self, channel_block):
        """
        Get channel names
        
        Input:
            - channel_block, string, block from summary containing the channel names
        
        Output
            - return, list, channel names
        """
        # Split by line
        lines = channel_block.split('\n')
        # Pattern for the channel
        pattern = re.compile("Channel [0-9]{1,}: (.*?)$")

        # Searching and appending all the channels
        output_channel_list = []
        for line in lines:
            channel_name = pattern.search(line)
            if channel_name is not None:
                channel_name = channel_name.group(1)
                output_channel_list.append(channel_name)

        return output_channel_list

    def _parse_metadata(self, metadata_block, output_metadata):
        """
        Parse a single metadata block with the seizure info
        Input:
            - metadata_block, list, all info related to a block
            - output_metadata, dict, all the block's information mapped and collected so far
        Output:
            - output_metadata, dict, after adding the new block
        """
        # Defining patterns
        pattern_filename = re.compile("File Name: (.*?)$")
        pattern_start_time = re.compile("File Start Time: (.*?)$")
        pattern_end_time = re.compile("File End Time: (.*?)$")
        pattern_seizures = re.compile("Number of Seizures in File: (.*?)$")
        pattern_seizure_start = re.compile("Seizure [0-9]{0,}[ ]{0,}Start Time: (.*?) seconds")
        pattern_seizure_end = re.compile("Seizure [0-9]{0,}[ ]{0,}End Time: (.*?) seconds")

        # Searching the patters for getting the info
        if pattern_filename.search(metadata_block[0]) is not None:
            file_metadata = dict()
            filename = pattern_filename.search(metadata_block[0]).group(1)
            file_metadata['start_time'] = pattern_start_time.search(metadata_block[1]).group(1)
            file_metadata['end_time'] = pattern_end_time.search(metadata_block[2]).group(1)
            file_metadata['n_seizures'] = int(pattern_seizures.search(metadata_block[3]).group(1))
            file_metadata['channel_names'] = self._channel_names
            
            # Inside a file, several seizures can be registered
            seizure_intervals = []
            for i in range(file_metadata['n_seizures']):
                seizure_start = int(pattern_seizure_start.search(metadata_block[4 + i * 2]).group(1))
                seizure_end = int(pattern_seizure_end.search(metadata_block[4 + i * 2 + 1]).group(1))
                seizure_intervals.append((seizure_start, seizure_end))
            file_metadata['seizure_intervals'] = seizure_intervals
            output_metadata[filename] = file_metadata
        else:
            # In some files, there are additional info apart from block's info
            warnings.warn("Block didn't follow the pattern for a metadata file block", Warning)
            # Check channel names
            try:
                self._channel_names = self._parse_channel_names("\n".join(metadata_block))
            except Exception as e:
                print ('Failed to parse block as a channel names block')
                raise e
        return output_metadata

    def _parse_file_metadata(self, seizure_file_blocks):
        """
        Parse the file metadata list blocks to get the seizure intervals
        *Note: These are not necessarily in file order, so always check against the filename before continuing.
        
        Input:
            - seizure_file_blocks, list, seizure file blocks
            
        Output:
            - output_metadata, dict, all seizure file block's information mapped and collected
        """
        output_metadata = OrderedDict()
        
        # Parsing and gathering the info for each block
        for block in seizure_file_blocks:
            lines = block.split('\n')
            output_metadata = self._parse_metadata(lines, output_metadata)
            
        return output_metadata

    def get_channel_names(self, filename):
        """
        Return the channel names
        """
        return self._metadata_store[filename]['channel_names']

    def get_seizure_list(self):
        """
        Get list of seizure intervals for each file
        """
        return [metadata['seizure_intervals'] for filename, metadata in self._metadata_store.items()]

    def get_file_metadata(self):
        """
        Get the metadata for all of the files
        """
        return self._metadata_store