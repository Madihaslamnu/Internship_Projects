This project covers data cleaning on the Olist e-commerce dataset in three parts: handling missing values by classifying why each column has gaps and 
choosing to drop, fill, or flag accordingly; finding and fixing duplicate rows and inconsistent city names using fuzzy matching, plus correcting wrong data 
types; and detecting price and freight outliers using both IQR and z-score methods, with manual inspection to decide what to keep. All logic is packaged into 
clean_data.py with a separate function per concern, plus a before/after data quality report.
