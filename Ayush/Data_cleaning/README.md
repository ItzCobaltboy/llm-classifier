# Data Cleaning and Transformation
Downlaod the `utils.py` and the corresponding python file to the transformation you want to apply on your data.
Keep the two files in the folder in which `Dataset` flder is there (adjecent to `Dataset` folder).
Run the python file.

### Following changes can be done using the geven python files
  - Removing outliers using the IQR formula.
  - Truncate the text to `MAX_LENGTH`.
  - Truncate the text to `MAX_LENGTH` or add a "<PAD>" word to match the `MAX_LENGTH`.
  - Split the text into multiple parts of `MAX_LENGTH`.
`MAX_LENGTH` is set to 500. To change it pass `df, max_length` instead of only `df`.

Additionaly you can either **upsample** or **downsample** the data to make no of samples foe each group equal.

If you want to remove noise, convert to lowercase or remove punctuation marks. Edit the python files and remove the `#` from commented lines.

You can make desired changes in functions in `utils.py` file.
