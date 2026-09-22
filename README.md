# PQM Readings Dashboard

A Streamlit app that accepts a PQM CSV file with 1-minute readings, extracts all numeric measurement columns, aggregates the values into 15-minute intervals, and plots the trend.

### How to run it on your own machine

1. Install the requirements

   ```bash
   pip install -r requirements.txt
   ```

2. Run the app

   ```bash
   streamlit run streamlit_app.py
   ```

### What the app expects

- A CSV file containing a timestamp/date column such as Timestamp, Datetime, DateTime, Date, or Time.
- Measurement columns containing numeric readings. All columns other than the timestamp column are checked and non-numeric or empty columns are ignored.
- Data recorded at 1-minute intervals, which will be resampled into 15-minute bins for plotting.

