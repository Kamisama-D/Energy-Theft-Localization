import os
import pandas as pd
import numpy as np

class Base_Load:
    """
    Represents the base load read from dataset.
    The class reads a specified column (by default "Total Kwh") from a designated sheet,
    and—if desired—reads all sheets (each representing a charging location) into a dictionary.

    Attributes:
      excel_file (str): Path to the Excel file.
      sheet_name (str): If read_all is False, name of the sheet to read.
      load_column (str): Name of the column representing load.
      df (DataFrame): DataFrame for the specified sheet.
      base_load_series (Series): Series of load values from the specified sheet.
      location_data (dict): Dictionary where keys are sheet names (locations) and values are DataFrames
                            containing the "Time" and "Kwh" columns.
    """
    def __init__(self,
                 excel_file:str = None,
                 sheet_name:str = "Total Load",
                 load_column:str = "Total Kwh",
                 read_all: bool = False):
        # If no excel_file is provided, compute the path relative to this file.
        if excel_file is None:
            # Get the directory of this file (Classes folder)
            base_dir = os.path.dirname(__file__)
            # Construct the path to the Dataset folder, one level up from Classes.
            excel_file = os.path.join(base_dir, "..", "Dataset", "Data.xlsx")
        self.excel_file = excel_file
        self.sheet_name = sheet_name
        self.load_column = load_column
        self.df = None
        self.base_load_series = None
        self.location_data = {} # a dictionary mapping location names to their DataFrame data.

        if read_all:
            self._read_all_data()
        else:
            self._read_data()

    def _read_data(self):
        self.df = pd.read_excel(self.excel_file, sheet_name=self.sheet_name)
        # Remove extra spaces from column names
        self.df.columns = self.df.columns.str.strip()
        if self.load_column not in self.df.columns:
            raise ValueError(f"Column '{self.load_column}' not found in sheet '{self.sheet_name}'.")
        self.base_load_series = self.df[self.load_column]

    def get_load(self, simulation_steps: int = None) -> np.ndarray:
        """
        Returns the base load time series as a numpy array.
        If simulation_steps is provided, only the first simulation_steps values are returned.
        """
        if simulation_steps is None:
            return self.base_load_series.to_numpy()
        else:
            return self.base_load_series.iloc[:simulation_steps].to_numpy()

    def get_time_axis(self, simulation_steps: int = None) -> np.ndarray:
        """
        If the sheet contains a "Time" column, return it as a numpy array of datetime objects.
        Otherwise, returns None.
        """
        if "Time" in self.df.columns:
            time_series = pd.to_datetime(self.df["Time"])
            if simulation_steps is not None:
                time_series = time_series.iloc[:simulation_steps]
            return time_series.to_numpy()
        else:
            return None

    def _read_all_data(self):
        """
       Reads all sheets from the Excel file into the location_data dictionary.
       Each key is a sheet name and each value is a DataFrame with the location’s data.
       If available, only the "Time" and "Kwh" columns are kept.
       """
        all_data = pd.read_excel(self.excel_file, sheet_name=None)
        for sheet_name, df in all_data.items():
            # Standardize column names by removing extra spaces and capitalizing.
            df.columns = df.columns.str.strip().str.capitalize()
            # If both "Time" and "Kwh" columns exist, convert Time to datetime and select these columns.
            if "Time" in df.columns and "Kwh" in df.columns:
                df["Time"] = pd.to_datetime(df["Time"])
                self.location_data[sheet_name] = df[["Time", "Kwh"]].copy()
            else:
                # Otherwise, store the entire DataFrame
                self.location_data[sheet_name] = df.copy()

    def get_location_data(self) -> dict:
        """
        Returns the dictionary containing the data for each location.
        Keys are location names (sheet names), and values are DataFrames
        that (preferably) contain the columns "Time" and "Kwh".
        """
        return self.location_data

if __name__ == '__main__':
    # Create an instance of Base_Load for 24 simulation hours.
    base_load_obj = Base_Load()
    # Print the base load series as a list.
    print("Aggregated base load series:", base_load_obj.base_load_series.tolist()[:97])

    # Create another instance of Base_Load for whole dataset
    base_load_yearly_obj = Base_Load(read_all=True)
    location_data = base_load_yearly_obj.get_location_data()
    for location, df in location_data.items():
        print(f"Location: {location}")
        print(df.head())
