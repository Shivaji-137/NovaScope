# NovaScope

NovaScope is a Streamlit dashboard I use to turn raw scientific datasets into clean, interactive plots within a couple of clicks. Drop in **CSV**, **TXT**, **DAT**, **NPZ**, **FITS**, or **HDF5** files, pick a plot style, and the visuals update immediately.

streamlit apps: [https://novascope-datavisualization.streamlit.app/](https://novascope-datavisualization.streamlit.app/)

## Why this exists

I work with a lot of data from different sources—simulations, telescope observations, and lab equipment. Every time I got a new file, I had to write custom code just to see a basic chart. This wasted time and slowed down my actual work. NovaScope fixes that problem: just drop in your file, look at your data, and pick the chart you want. No coding required unless you want to customize something.

## Highlights

- **Easy file upload** — Just drag and drop your data file
- **Smart previews** — Browse multiple datasets within one file, view FITS images with color controls
- **Many chart types** — Line charts, bar charts, scatter plots, histograms, heatmaps, 3D plots, and more
- **Simple controls** — Pick columns from the sidebar or click directly in the data table
- **Extra features** — Group by color, use log scales, calculate averages, and more
- **Data insights** — See statistics and preview your data before plotting

## Supported file types

| Extension              | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `.csv`, `.txt`, `.dat` | Delimited text with automatic separator detection        |
| `.npz`                 | NumPy archives (arrays become DataFrame columns)         |
| `.fits`                | FITS tables + image HDUs via **Astropy**                 |
| `.hdf5`                | Hierarchical datasets via **h5py**                       |

## Plot catalog at a glance

| Family                    | Chart types                                   |
| ------------------------- | --------------------------------------------- |
| Trend & temporal          | line, area                                    |
| Comparisons               | scatter, bar, violin, pie                     |
| Distributions             | histogram, KDE, PDF, ECDF                     |
| Correlation & density     | heatmap, density heatmap, contour, regression |
| Multivariate              | scatter matrix, pairplot                      |
| Scientific                | 3D scatter (with optional size encoding)      |

## Getting started

1. *(Optional)* Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

4. Upload a dataset, pick a plot family, and tune the options in the sidebar. Charts refresh automatically.

## Project structure

```
NovaScope/
├── app.py                 # Streamlit UI + layout
├── plotter/
│   ├── __init__.py
│   ├── data_loader.py     # Format-specific loaders (CSV, NPZ, FITS, HDF5…)
│   └── plotting.py        # Plot catalog + rendering helpers
├── tests/
│   ├── test_data_loader.py
│   └── test_plotting.py
├── requirements.txt
└── README.md
```

## Development notes

Every loader, UI workflow, and plotting hook started from a blank file—I wrote the application logic end to end with the help of existing library such as astropy, h5py, pandas, seaborn, etc so I could understand every data path and keep the project lightweight. When it was time to optimize performance and polish the CSS shell, I leaned on AI assistance for ideas and quick iterations. 

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues for bug reports, feature suggestions, or submit pull requests.

## Author

- **Name:** Shivaji Chaulagain
- **Email:** shivajichaulagain@gmail.com
- **Website:** [www.shivajichaulagain.com.np](https://www.shivajichaulagain.com.np)

## Happy plotting!


