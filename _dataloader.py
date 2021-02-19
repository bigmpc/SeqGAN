import numpy as np


class DataLoader:
    """DataLoader Class Load data and iterable 
        Note:
            - To create new instance of class with file ptath use  DataLoader.from_txt_file(file_path ,...)
            - You can for loop to iterate on sequence_batch
            
        Attributes:
            sequence_batch (numpy.ndarray): numpy array
            num_batch (int): number of batch .
            pointer (int): pointer for __iter__ function
        """

    def __init__(self, sequence_batch: np.ndarray, num_batch: int):
        """create new instance class

        Parameters
        ----------
        sequence_batch : numpy.ndarray
            batch of data
        num_batch : int
            number of batchs
        """

        self.sequence_batch = sequence_batch
        self.num_batch = num_batch

    @classmethod
    def from_txt_file(cls, file_path: str, batch_size: int = 64, token_size: int = 20) -> DataLoader:
        """Read .txt file and create new DataLoader class

        Parameters
        ----------
        cls : DataLoader

        file_path : str
            A flag used to print the columns to the console (default is False)
        batch_size : int
            batch size of dataset
        token_size : int
            size of valid token
      
        Returns
        -------
        DataLoader
            create new instance DataLoader class
        """

        token_stream = [sample for sample in np.loadtxt(
            file_path) if len(sample) == token_size]
        num_batch = len(token_stream) // batch_size
        sequence_batch = np.split(np.array(token_stream), num_batch, 0)
        return cls(sequence_batch, num_batch)

    def __iter__(self):
        """create iterable class and set pointer to 0

        Parameters
        ----------
        self : class
            self class variable
        
        Returns
        -------
        self
            self class variable
        """

        self.pointer = 0
        return self

    def __next__(self) -> np.ndarray:
        """get next batch data and return it

        Parameters
        ----------
        self : class
            self class variable

        Returns
        -------
        next_batch: np.ndarray
            next batch data
            
        Raise
        -------
        StopIteration
            when get end of dataset then raise StopIteration to stop loop
        """

        if self.pointer >= self.num_batch:
            raise StopIteration
        next_batch = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1)
        return next_batch


Gen_Data_loader = DataLoader
Dis_dataloader = DataLoader
