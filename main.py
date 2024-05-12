import os
from bs4 import BeautifulSoup
import re
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans    
from sklearn.metrics import silhouette_score, pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_hex
import networkx as nx
import numpy as np
import community as community_louvain
import seaborn as sns
import warnings
from scipy.cluster.hierarchy import ClusterWarning

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Suppress ClusterWarning
warnings.filterwarnings("ignore", category=ClusterWarning)

# Suppress UserWarning
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")

#plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

# Count words in a tag
def count_words(tag):
    ab_tag = tag.find('ab') # Find the 'ab' tag
    if ab_tag: # If the 'ab' tag exists
        return len(re.findall(r'\S+', ab_tag.get_text())) # Count the words
    return 0 # Return 0 if the 'ab' tag doesn't exist

# Process XML file and generate CSV files for character interactions and characters on stage
def process_xml(xml_file):
    # # Read and Parse the XML content using BeautifulSoup
    with open(xml_file, 'r') as file: # Open the XML file
        content = file.read() # Read the content of the XML file
    soup = BeautifulSoup(content, 'xml') # Parse the content of the XML file using BeautifulSoup

    # Initialize current narrative time, on_stage list and a dictionary to record characters on stage at each time point
    current_time = "0" # Initialize current narrative time
    on_stage = [] # Initialize on_stage list
    stage_timeline = {} # {current_time: [characters_on_stage]}
    character_exhange = {} # {speaker: {listener: word_count}}

    # Iterate through all nodes in the XML file
    for tag in soup.find_all(['sp', 'stage', 'milestone']):
        # Update current time from milestone using xml:id attribute
        if tag.name == 'milestone' and tag.get('unit') == 'ftln':
            current_time = tag['xml:id'] # Update current time
            # Record characters on stage for the current time
            stage_timeline[current_time] = on_stage.copy() # Use copy to avoid reference issues

        # Stage direction for entrance
        if tag.name == 'stage' and tag.get('type') == 'entrance' and 'who' in tag.attrs:
            characters_entering = [re.sub(r"_[^\s,]*", "", char.replace("#", "")) for char in tag['who'].split()]
            on_stage = list(set(on_stage + characters_entering))
        
        # Stage direction for exit
        if tag.name == 'stage' and tag.get('type') == 'exit' and 'who' in tag.attrs:
            characters_exiting = [re.sub(r"_[^\s,]*", "", char.replace("#", "")) for char in tag['who'].split()]
            on_stage = [char for char in on_stage if char not in characters_exiting]
        # Speech tag 
        if tag.name == 'sp' and 'who' in tag.attrs: # Check if the tag is a speech
            speakers = [re.sub(r"_[^\s,]*", "", speaker.replace("#", "")) for speaker in tag['who'].split()]
            for speaker in speakers:  # Loop through each speaker
                others = [char for char in on_stage if char != speaker]  # Listeners for this specific speaker
                if not others: # If there are no listeners, skip to the next speaker
                    others.append(speaker) # Add the speaker to the list of listeners
                
                word_count = count_words(tag) // len(speakers)  # Divide the word count among the speakers
                
                for listener in others: # Loop through each listener
                    character_exhange.setdefault(speaker, {}).setdefault(listener, 0) # Add speaker and listener to the dictionary if they don't exist
                    character_exhange[speaker][listener] += word_count # Add the word count to the speaker-listener pair
    
    # Convert interactions dictionary to dataframe and transpose
    character_exhange_df = pd.DataFrame(character_exhange).T.fillna(0) # Convert the dictionary to a dataframe and transpose it

    all_characters = list(set(character_exhange_df.index).union(set(character_exhange_df.columns))) # Get all characters
    character_exhange_df = character_exhange_df.reindex(index=all_characters, columns=all_characters).fillna(0) # Reindex the dataframe and fill missing values with 0
    total_words = character_exhange_df.sum(axis=1) # Sum the words spoken by each character
    sorted_characters = total_words.sort_values(ascending=False).index.tolist()
    character_exhange_df = character_exhange_df.loc[sorted_characters, sorted_characters]

    # Create a set of all unique characters that ever appear on stage
    all_characters = set()
    for char_list in stage_timeline.values():
        for char in char_list:
            all_characters.add(char)

    # Convert the set to a sorted list
    all_characters = sorted(list(all_characters))

    # Create an empty DataFrame with rows of all characters and columns of all the current_time values
    stage_timeline_df = pd.DataFrame(index=all_characters, columns=stage_timeline.keys(), dtype=int)

    # Fill in the DataFrame
    for current_time, characters_on_stage in stage_timeline.items():
        for char in all_characters:
            if char in characters_on_stage:
                stage_timeline_df.loc[char, current_time] = 1
            else:
                stage_timeline_df.loc[char, current_time] = 0

    # Save the exhange DataFrame to a CSV file
    exchange_output_dir = "output_exchange" # Directory to save the CSV files
    if not os.path.exists(exchange_output_dir):
        os.makedirs(exchange_output_dir)

    exchange_output_filename = os.path.basename(xml_file).replace(".xml", "_exchange.csv")
    exchnage_output_path = os.path.join(exchange_output_dir, exchange_output_filename)
    character_exhange_df.to_csv(exchnage_output_path, index=True)
    
    # Save the onstage DataFrame to CSV file
    onstage_output_directory = "output_onstage"
    if not os.path.exists(onstage_output_directory):
        os.makedirs(onstage_output_directory)

    onstage_output_file_name = os.path.basename(xml_file).replace(".xml", "_onstage.csv")
    onstage_output_file_path = os.path.join(onstage_output_directory, onstage_output_file_name)
    stage_timeline_df.to_csv(onstage_output_file_path)
    
    # print(f"Processed {xml_file}.")
    # print(f"The generated exchange csv file has {character_exhange_df.shape[0]} rows and {character_exhange_df.shape[1]} columns.")
    # print(f"The generated onstage CSV file has {stage_timeline_df.shape[0]} rows and {stage_timeline_df.shape[1]} columns.")

    return stage_timeline_df, character_exhange_df

# Calculate the separation score
def calculate_separation_score(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Drop the first column if it's just character names or identifiers
    if data.columns[0] != 'ftln':
        data = data.drop(data.columns[0], axis=1)
    
    # Compute the similarity matrix
    normalized_similarity_matrix = data.T.dot(data) / data.T.dot(data).max().max()
    
    # Compute a distance matrix from the similarity matrix
    distance_matrix = 1 - normalized_similarity_matrix
    
    # Perform hierarchical clustering
    Z = linkage(pairwise_distances(data.T), method='average')
    
    # Determine the optimal number of clusters (aiming for 2) and assign characters to clusters
    clusters = fcluster(Z, t=2, criterion='maxclust')
    
    # Calculate the separation score
    cluster_distribution = pd.Series(clusters).value_counts()
    separation_score = cluster_distribution.min() / cluster_distribution.sum()
    
    print(f"Separation score for {file_path}: {separation_score}")
    return separation_score

# Generate heatmap for characters on stage
def onstage_heatmap(onstage_df, xml_file):
    # Modify x labels
    x_labels = [label.replace("ftln-", "TLN ") for label in onstage_df.columns]
    
    # # Extract the suffix from filename for y labels modification
    # file_suffix = "_" + os.path.basename(xml_file).replace(".xml", "")
    
    # Modify y labels
    y_labels = [label.title() for label in onstage_df.index]

    # Data for heatmap
    z = onstage_df.values
    x = x_labels
    y = y_labels

    hovertext = []

    for y_label, row in zip(y_labels, z):
        hover_row = []
        for x_label, val in zip(x_labels, row):
            tln = x_label.split()[-1]  # Assuming the format is 'TLN xxx'
            presence = 'Yes' if val == 1 else 'No'
            hover_data = f"Through Line Number: {tln}<br>Character: {y_label}<br>Presence: {presence}"
            hover_row.append(hover_data)
        hovertext.append(hover_row)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        hovertext=hovertext,
        hoverinfo='text',  # Use custom hovertext
        colorscale='Blues',
        showscale=False
    ))

    # Update the layout
    fig.update_layout(
        autosize=False,
        width=10000,
        height=len(onstage_df.index) * 30,  # Adjusting height based on number of characters
        margin=dict(t=50, r=50, b=100, l=200),  # Adjust margins to fit labels
        yaxis=dict(tickangle=-30),  # Adjust y-axis tick angle for better readability
    )
    output_directory = "output_onstage_heatmap"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Get the output filename based on the mapping
    output_file_name = os.path.basename(xml_file).replace(".xml", "_onstage_heatmap.html")
    output_file_path = os.path.join(output_directory, output_file_name)
    fig.write_html(output_file_path)

    # Generate .png file
    plt.figure(figsize=(20, 10))
    sns.heatmap(z, xticklabels=x, yticklabels=y, cmap='Blues', annot=False, cbar=False)
    

    # Obtain colors from the colormap
    cmap = plt.cm.Blues
    onstage_color = to_hex(cmap(0.8))  # Dark blue, typically indicates higher values or "Onstage"
    offstage_color = to_hex(cmap(0.1))  # Light blue, near white, for lower values or "Offstage"

    # Custom legend elements
    legend_elements = [
        Patch(facecolor=onstage_color, edgecolor='black', label='Onstage'),
        Patch(facecolor=offstage_color, edgecolor='black', label='Offstage')
    ]

    # Add legend to the plot
    plt.legend(handles=legend_elements, loc='lower left')

    # Adjust x-tick intervals
    interval = 100
    minor_interval = 50
    # Calculate x-tick positions to ensure the last label is included
    tick_positions = np.arange(0, len(x), interval)
    # Adjust to include the last label if it doesn't fall on an exact multiple of interval
    if tick_positions[-1] != len(x) - 1:
        tick_positions = np.append(tick_positions, len(x) - 1)

    # Calculate positions for minor ticks
    minor_tick_positions = np.arange(0, len(x), minor_interval)

    # Make sure to grab labels correctly, ensuring the last label is included
    tick_labels = [x[pos-1] if pos > 0 else x[0] for pos in tick_positions]
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=90, ha='right')

    # Set minor ticks without labels
    plt.gca().set_xticks(minor_tick_positions, minor=True)
    plt.gca().tick_params(axis='x', which='minor', length=4)  # Customize minor tick length and appearance


    plt.yticks(rotation=0)
    
    #title_text = r"Onstage Characters in \textit{" + os.path.splitext(os.path.basename(xml_file))[0] + "}"
    title_text = "Onstage Characters in " + os.path.splitext(os.path.basename(xml_file))[0]
    plt.title(title_text)
    plt.xlabel("Through Line Number (TLN)")
    plt.ylabel("Character")

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, os.path.splitext(output_file_name)[0] + '.png'), dpi=600)
    plt.close()

# Generate heatmap for character interactions
def exchange_heatmap(exchange_df, xml_file):

    character_names = exchange_df.columns[0:].tolist() 

    # Heatmap Visualization with Plotly
    colorscale = [
            [0, 'rgb(255, 255, 255)'],       # color for value 0 (white)
            [1e-9, 'rgb(173, 216, 230)'],    # color just above 0 (light blue)
            [500/6000, 'rgb(100, 149, 237)'], # color around 500 (a darker shade of blue)
            [1, 'rgb(0, 0, 139)']            # color for the max value, 6000 (darkest blue)
        ]
    heatmap_fig = go.Figure(data=go.Heatmap(
                        z=np.flipud(exchange_df), # Flip the matrix vertically
                        x=character_names,
                        y=list(reversed(character_names)),
                        colorscale=colorscale,
                        zmin=0,
                        zmax=6000,
                        hovertemplate='Speaker: %{y}<br>Listener: %{x}<br>Words: %{z}<extra></extra>'
                    ))

    heatmap_fig.update_layout(
        title=f"Heatmap of Interactions in {os.path.splitext(xml_file)[0]}",
        xaxis=dict(side='top'),
        yaxis=dict(autorange='reversed')  # Ensure the y-axis is correctly flipped
        )

     # Generate the output filename
    output_filename = os.path.splitext(os.path.basename(xml_file))[0] + '_exchange_heatmap.html'

    output_directory = "output_exchange_heatmap"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Write the heatmap to an HTML file using the generated filename
    heatmap_fig.write_html(os.path.join(output_directory, output_filename)) 

    # Generate .png file
    plt.figure(figsize=(20, 10))
    sns.heatmap(exchange_df, xticklabels=character_names, yticklabels=character_names, cmap='Blues', annot=False, cbar=False)
    # Show x-axis label vertically
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)

    # Show values as interger in the heatmap. Do not show zero values.
    for i in range(exchange_df.shape[0]):
        for j in range(exchange_df.shape[1]):
            if exchange_df.iloc[i, j] != 0:
                plt.text(j + 0.5, i + 0.5, int(exchange_df.iloc[i, j]), ha='center', va='center', color='white')


    #title_text = r"Character Interactions in \textit{" + os.path.splitext(os.path.basename(xml_file))[0] + "}"
    title_text = "The Number of Words Exchanged between Characters in " + os.path.splitext(os.path.basename(xml_file))[0]
    plt.title(title_text)
    # Title for x-axis: Character (Listener)
    plt.xlabel("Character (Listener)")
    
    # Title for y-axis: Character (Speaker)
    plt.ylabel("Character (Speaker)")

    # plt.title(f"Character Interactions in {os.path.splitext(os.path.basename(xml_file))[0]}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, os.path.splitext(output_filename)[0] + '.png'), dpi=600)
    plt.close()


# Generate co-occurrence matrix
def onstage_matrix(onstae_df, xml_file):

    # Compute the concurrence matrix by multiplying the binary presence matrix with its transpose
    cooccurrence_matrix = onstage_df.dot(onstage_df.T)

    # Adjust the diagonal to 0 to exclude self-concurrences
    np.fill_diagonal(cooccurrence_matrix.values, 0)
    
    # Save the concurrence matrix to a CSV file
    cooccurrence_output_directory = "output_onstage_matrix"
    if not os.path.exists(cooccurrence_output_directory):
        os.makedirs(cooccurrence_output_directory)
    
    cooccurrence_output_filename = os.path.basename(xml_file).replace(".xml", "_onstage_matrix.csv")
    cooccurrence_output_path = os.path.join(cooccurrence_output_directory, cooccurrence_output_filename)
    cooccurrence_matrix.to_csv(cooccurrence_output_path, index=True)

    return cooccurrence_matrix

# Calculate Dunn Index for clustering
def dunn_index(points, labels):
    """
    Calculate Dunn Index for the given clustering
    """
    distances = pairwise_distances(points)
    intra_cluster_distances = [np.mean(distances[labels == label]) for label in np.unique(labels)]
    min_inter_cluster_distance = np.min([np.min(distances[labels == label_i, :][:, labels == label_j])
                                         for i, label_i in enumerate(np.unique(labels))
                                         for j, label_j in enumerate(np.unique(labels)) if i != j])
    return min_inter_cluster_distance / max(intra_cluster_distances)

def degree_centralization(G):
    degrees = nx.degree(G)
    max_degree = max(dict(degrees).values())
    total = sum(max_degree - degree for node, degree in degrees)
    # For a star network, max possible total is (n-1)*(n-2) for n nodes
    n = len(G.nodes)
    max_total = (n - 1) * (n - 2)
    return total / max_total if n > 1 else 0

def betweenness_centralization(G):
    betweenness = nx.betweenness_centrality(G)
    max_betweenness = max(betweenness.values())
    total = sum(max_betweenness - b for b in betweenness.values())
    # For a star network, max possible total is (n-1)*(n-2) for n nodes
    n = len(G.nodes)
    max_total = (n - 1) * (n - 2) / 2
    return total / max_total if n > 1 else 0

def closeness_centralization(G):
    closeness = nx.closeness_centrality(G)
    max_closeness = max(closeness.values())
    total = sum(max_closeness - c for c in closeness.values())
    # Normalization factor for closeness centralization varies, 
    # so we use a more complex formula depending on graph connectivity
    n = len(G.nodes)
    if nx.is_connected(G):
        denom = (n-1) * (n-2) / (2*n-3) if n > 2 else 1
    else:
        denom = (n-1) ** 2 / (n-2)
    return total / denom if n > 1 else 0

def eigenvector_centralization(G):
    eigenvector = nx.eigenvector_centrality(G)
    max_eigenvector = max(eigenvector.values())
    total = sum(max_eigenvector - e for e in eigenvector.values())
    # For eigenvector centralization, normalization also involves complex considerations,
    # generally using the sum of squares of differences in eigenvector centralities
    # but here we simplify it for demonstration
    n = len(G.nodes)
    max_total = (n - 1) ** 2  # Simplified approximation
    return total / max_total if n > 1 else 0

# Visualize network
def visualize_network(csv_file_path, output_file_name, kmeans=None, community_detection=True):
    # Extract the .csv filename to use as the plot title
    filename = os.path.basename(csv_file_path)
    plot_title = filename.replace(".csv", "").replace("_", " ") + " Network" # Use the filename as the plot title.
    
    # Remove matrix from the title
    if " matrix" in plot_title.lower():
        plot_title = plot_title.replace(" matrix", "")

    # Load the CSV file as a pandas DataFrame
    data = pd.read_csv(csv_file_path)
    
    # Create a graph
    G = nx.Graph()
    
    # Nodes and edges
    nodes = data.columns[1:].tolist()  # Assuming the first column is for labeling and the rest are nodes
    for i, row in data.iterrows():
        for j, col in enumerate(data.columns[1:]):  # Skipping the first column
            weight = data.iloc[i, j+1]  # Adjusting index for weight
            if weight > 0:
                G.add_edge(row[data.columns[0]], col, weight=weight)
    
    # Compute the spring layout
    pos = nx.spring_layout(G)
    
    # Community detection
    partition = community_louvain.best_partition(G) if community_detection else {}
    communities = list(partition.values()) if community_detection else [0] * len(G.nodes())
    
    # KMeans clustering
    if kmeans is not None:
        pos_array = np.array(list(pos.values()))
        kmeans_result = KMeans(n_clusters=kmeans, n_init=10).fit(pos_array)
        clusters = kmeans_result.labels_
        silhouette = silhouette_score(pos_array, clusters)
        dunn = dunn_index(pos_array, clusters)
    else:
        clusters = [0] * len(G.nodes())
        silhouette, dunn = None, None
    
    # Calculate degree centralization for the entire graph
    degree_cent = degree_centralization(G)

    # Calculate betweenness centralization for the entire graph
    betweenness_cent = betweenness_centralization(G)

    # Calculate closeness centralization for the entire graph
    closeness_cent = closeness_centralization(G)

    # Calculate eigenvector centralization for the entire graph
    eigenvector_cent = eigenvector_centralization(G)
    
    # Edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    
    # Node traces and annotations for labels
    node_traces = []
    annotations = []
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'pentagon', 'hexagon', 'octagon']
    for cluster_label in set(clusters):
        x, y, community_colors, texts = [], [], [], []
        for node, cluster in zip(G.nodes(), clusters):
            if cluster == cluster_label:
                node_x, node_y = pos[node]
                x.append(node_x)
                y.append(node_y)
                community_id = partition.get(node, "N/A")
                cluster_id = cluster
                node_label = f'{node}'
                hover_text = f'Node: {node}<br>Community: {community_id}<br>Cluster: {cluster_id}'
                community_colors.append(partition.get(node, 0))
                texts.append(hover_text)
                
                annotations.append(dict(x=node_x, y=node_y+0.02,
                                        text=node_label,
                                        showarrow=False,
                                        xanchor='center',
                                        yanchor='bottom',
                                        font=dict(family='Arial', size=12)))
        
        symbol = symbols[cluster_label % len(symbols)]
        node_trace = go.Scatter(x=x, y=y, mode='markers', hoverinfo='text', text=texts,
                                marker=dict(symbol=symbol, size=10, color=community_colors, colorscale='Viridis',
                                            line=dict(color='black', width=1)))
        node_traces.append(node_trace)
    
    # Combine traces and configure layout with annotations
    fig = go.Figure(data=[edge_trace, *node_traces],
                    layout=go.Layout(title_text=plot_title.title(),  # Set the extracted filename as the plot title
                                     showlegend=False, hovermode='closest',
                                     margin=dict(b=0, l=0, r=0, t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     annotations=annotations))
    
    # Save the figure to an HTML file
    fig.write_html(output_file_name)


    # Save the figure as .png file
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_color=communities, node_size=300, cmap=plt.cm.tab20, with_labels=True)
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(output_file_name.replace(".html", ".png"), dpi=600)
    plt.close()

    return silhouette, dunn, degree_cent, betweenness_cent, closeness_cent, eigenvector_cent


# Plot scores (original)
def plot_scores(scores_dict, plot_title, ylabel, output_file_name, plot_type='bar', sort='score'):
    # Output directory
    output_directory = "output_scores"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Extract the filenames from the scores_dict
    filenames = [os.path.basename(file) for file in scores_dict.keys()]

    # Remove the file extension from the filenames
    filenames = [os.path.splitext(file)[0] for file in filenames]

    # Remove the suffix from the filenames
    filenames = [re.sub(r"_[^\s,]*", "", file) for file in filenames]

    scores = list(scores_dict.values())

    # Open metadata file to get Title, Year, and Genre
    with open('metadata.csv', 'r') as file:
        metadata = pd.read_csv(file)
    
    # Initialize a score column to 0
    metadata['Score'] = 0.0
    
    # Add scores to metadata based on matching titles
    for i, row in metadata.iterrows():
        title = row['Title']
        for j, filename in enumerate(filenames):
            if title.lower() in filename.lower():
                metadata.at[i, 'Score'] = scores[j]
    
    # Create a unique identifier for each title by combining Title and Year
    metadata['UniqueTitle'] = metadata.apply(lambda row: f"{row['Title']} ({row['Year']})", axis=1)
    
    # Handle NaN values by replacing them with a placeholder, e.g., 'NA'
    metadata['Genre'].fillna('NA', inplace=True)
    
    # Sort the metadata based on the chosen sort key
    if sort == 'score':
        sorted_metadata = metadata.sort_values('Score', ascending=False)
    elif sort == 'genre':
        sorted_metadata = metadata.sort_values('Genre')
    elif sort == 'year':
        sorted_metadata = metadata.sort_values('Year', ascending=True)
    else:
        sorted_metadata = metadata
        
    # Save the sorted_metadata to a CSV file with the plot_title as the filename
    sorted_metadata.to_csv(os.path.join(output_directory, f"{plot_title}.csv"), index=False)
    
    # Generate hover text after sorting to ensure correct order
    hover_text = sorted_metadata.apply(lambda row: f"Title: {row['Title']}<br>Year: {row['Year']}<br>Genre: {row['Genre']}<br>{ylabel}: {row['Score']}", axis=1).tolist()

    # Determine the type of plot
    if plot_type == 'bar':
        colors = {'Comedy': '#add8e6', 'Tragedy': '#708090', 'History': '#ff0000', 'NA': '#d3d3d3'}
        bar_colors = [colors[Genre] for Genre in sorted_metadata['Genre']]
        fig = go.Figure(go.Bar(x=sorted_metadata['UniqueTitle'], y=sorted_metadata['Score'], hovertext=hover_text, hoverinfo='text'))
        fig.update_traces(marker_color=bar_colors)
    elif plot_type == 'line':
        fig = go.Figure(go.Scatter(x=sorted_metadata['UniqueTitle'], y=sorted_metadata['Score'], mode='lines+markers', hovertext=hover_text, hoverinfo='text'))

    # Update the layout
    #fig.update_layout(title=plot_title, xaxis_title='Play Title', yaxis_title=ylabel, showlegend=True)
    fig.update_layout(title=plot_title, xaxis_title='Play Title', yaxis_title=ylabel, showlegend=False)

    # Show the legend top left inside the plot area
    fig.update_layout(legend=dict(title='Genre', orientation='h', yanchor='top', y=0.9, xanchor='left', x=0.01))

    # Save the figure to an HTML file in the output directory
    fig.write_html(os.path.join(output_directory, f"{output_file_name}.html"))
    
    # Static Plotting
    plt.figure(figsize=(10, 6))
    if plot_type == 'bar':
        sns.barplot(x='UniqueTitle', y='Score', hue='Genre', data=sorted_metadata, dodge=False, palette=colors)
        plt.xticks(rotation=45, ha='right')
    elif plot_type == 'line':
        sns.lineplot(x='UniqueTitle', y='Score', hue='Genre', data=sorted_metadata, marker='o', palette=colors)
        plt.xticks(rotation=45, ha='right')

    plt.title(plot_title)
    plt.xlabel('Play Title')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the figure to a PNG file in the output directory
    plt.savefig(os.path.join(output_directory, f"{output_file_name}.png"), dpi=600)
    plt.close()


    # Box plot generation for scores by genre with titles as dots    
    genres = sorted_metadata['Genre'].unique()
    
    fig = go.Figure()

    for genre in genres:
        genre_data = metadata[metadata['Genre'] == genre]
        genre_scores = genre_data['Score']
        if not genre_scores.empty:
            # Box plot for the genre
            fig.add_trace(go.Box(y=genre_scores, name=genre, boxpoints=False, boxmean=True, marker=dict(color='blue')))
            
            # Scatter plot for individual titles/scores as dots
            fig.add_trace(go.Scatter(
                x=[genre] * len(genre_scores),  # Align dots with the corresponding genre box
                y=genre_scores,
                mode='markers+text',
                marker=dict(color='red', size=8),  # Adjusted color to red and size to bigger
                hoverinfo='text',
                hovertext=genre_data['UniqueTitle'],
                text=genre_data['UniqueTitle'],
                textposition='top center',
                textfont=dict(size=10),  # Adjust text size as needed
                name=genre  # To ensure dots are aligned with the correct box plot
            ))
    
    fig.update_layout(title=f"{plot_title} Distribution by Genre", xaxis_title='Genre', yaxis_title=ylabel, showlegend=False)
    
    # Save the figure to an HTML file in the output directory
    fig.write_html(os.path.join(output_directory, f"{output_file_name}_genre.html"))

    # Static Plotting
    
    # Set plot size
    plt.figure(figsize=(12, 8))

    # Create box plot
    sns.boxplot(x='Genre', y='Score', data=sorted_metadata, palette='Set3')

    # Overlay scatter plot for individual scores
    # We use 'stripplot' for individual scores to align them with the boxplot but with jitter to avoid overlapping
    sns.stripplot(x='Genre', y='Score', data=sorted_metadata, color='black', jitter=0.1, size=5)

    # Add text annotations for each point placing them left to the scatter points
    for i, row in sorted_metadata.iterrows():
        plt.text(x=np.where(sorted_metadata['Genre'].unique() == row['Genre'])[0][0], 
                y=row['Score'], 
                s=row['UniqueTitle'],
                ha='center', va='bottom', fontsize=5)

    # Set titles and labels
    plt.title(plot_title + ' Distribution by Genre')
    plt.xlabel('Genre')
    plt.ylabel(ylabel)

    # Improve layout
    plt.tight_layout()

    # Save the figure
    output_directory = "output_scores"  # Ensure this directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    plt.savefig(os.path.join(output_directory, f"{output_file_name}_genre.png"), dpi=300)
    plt.close()


# List all .xml files in the 'Data' directory
data_dir = 'Data'
xml_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.xml')]

# # Process each XML file and generate CSV files for character interactions and characters on stage
for xml_file in xml_files:
    onstage_df, exchange_df = process_xml(xml_file)
    onstage_heatmap(onstage_df, xml_file)
    exchange_heatmap(exchange_df, xml_file)
    onstage_matrix(onstage_df, xml_file)
    print(f"Processed {xml_file}.")

# List all .csv files in the 'output_onstage' directory
on_stage_csv = [os.path.join('output_onstage', file) for file in os.listdir('output_onstage') if file.endswith('.csv')]

# List all .csv files in the 'output_exchange' directory
on_stage_matrix_csv = [os.path.join('output_onstage_matrix', file) for file in os.listdir('output_onstage_matrix') if file.endswith('.csv')]

# List all .csv files in the 'output_exchange' directory
exchange_csv = [os.path.join('output_exchange', file) for file in os.listdir('output_exchange') if file.endswith('.csv')]

# A dictionary to store the onstage silhouette scores for each CSV file
silhouette_scores_onstage = {} # {csv_file: silhouette_score}

# a dictionary to store the onstage Dunn Index for each CSV file
dunn_indices_onstage = {} # {csv_file: dunn_index}

# A dictionary to store the degree centralization scores for each CSV file
degree_centralization_onstage = {} # {csv_file: degree_centrality}

# A dictionary to store the betweenness centrality scores for each CSV file
betweenness_centrality_onstage = {} # {csv_file: betweenness_centrality}

# A dictionary to store the closeness centrality scores for each CSV file
closeness_centrality_onstage = {} # {csv_file: closeness_centrality}

# A dictionary to store the eigenvector centrality scores for each CSV file
eigenvector_centrality_onstage = {} # {csv_file: eigenvector_centrality}

# Visualize network for each CSV file in the 'output_onstage_matrix' directory
for csv_file in on_stage_matrix_csv:
    silhouette, dunn, degree_cent, betweenness_cent, closeness_cent, eigenvector_cent = visualize_network(csv_file, f"{os.path.splitext(csv_file)[0]}_network.html", kmeans=2, community_detection=True)
    silhouette_scores_onstage[csv_file] = silhouette
    dunn_indices_onstage[csv_file] = dunn
    degree_centralization_onstage[csv_file] = degree_cent
    betweenness_centrality_onstage[csv_file] = betweenness_cent
    closeness_centrality_onstage[csv_file] = closeness_cent
    eigenvector_centrality_onstage[csv_file] = eigenvector_cent
    print(f"Processed {csv_file}.")

    

# A dictionary to store the exchange silhouette scores for each CSV file
silhouette_scores_exchange = {} # {csv_file: silhouette_score}

# A dictionary to store the exchange Dunn Index for each CSV file
dunn_indices_exchange = {} # {csv_file: dunn_index}

# A dictionary to store the degree centralization scores for each CSV file
degree_centralization_exchange = {} # {csv_file: degree_centrality}

# A dictionary to store the betweenness centrality scores for each CSV file
betweenness_centrality_exchange = {} # {csv_file: betweenness_centrality}

# A dictionary to store the closeness centrality scores for each CSV file
closeness_centrality_exchange = {} # {csv_file: closeness_centrality}

# A dictionary to store the eigenvector centrality scores for each CSV file
eigenvector_centrality_exchange = {} # {csv_file: eigenvector_centrality}

# Visualize network for each CSV file in the 'output_exchange' directory
for csv_file in exchange_csv:
    silhouette, dunn, degree_cent, betweenness_cent, closeness_cent, eigenvector_cent = visualize_network(csv_file, f"{os.path.splitext(csv_file)[0]}_network.html", kmeans=2, community_detection=True)
    silhouette_scores_exchange[csv_file] = silhouette
    dunn_indices_exchange[csv_file] = dunn
    degree_centralization_exchange[csv_file] = degree_cent
    betweenness_centrality_exchange[csv_file] = betweenness_cent
    closeness_centrality_exchange[csv_file] = closeness_cent
    eigenvector_centrality_exchange[csv_file] = eigenvector_cent
    print(f"Processed {csv_file}.")

# Plot the silhouette scores for the onstage matrices
plot_scores(silhouette_scores_onstage, 'Silhouette Scores for Onstage Matrices', 'Silhouette Score', 'onstage_silhouette_scores', plot_type='bar', sort='year')

# Plot the Dunn Index for the onstage matrices
plot_scores(dunn_indices_onstage, 'Dunn Index for Onstage Matrices', 'Dunn Index', 'onstage_dunn_indices', plot_type='bar', sort='year')

# Plot the silhouette scores for the exchange matrices
plot_scores(silhouette_scores_exchange, 'Silhouette Scores for Exchange Matrices', 'Silhouette Score', 'exchange_silhouette_scores', plot_type='bar', sort='year')

# Plot the Dunn Index for the exchange matrices
plot_scores(dunn_indices_exchange, 'Dunn Index for Exchange Matrices', 'Dunn Index', 'exchange_dunn_indices', plot_type='bar', sort='year')

# Plot the degree centralization scores for the onstage matrices
plot_scores(degree_centralization_onstage, 'Degree Centralization Scores for Onstage Matrices', 'Degree Centralization', 'onstage_degree_centralization', plot_type='bar', sort='year')

# Plot the betweenness centrality scores for the onstage matrices
plot_scores(betweenness_centrality_onstage, 'Betweenness Centrality Scores for Onstage Matrices', 'Betweenness Centrality', 'onstage_betweenness_centrality', plot_type='bar', sort='year')

# Plot the closeness centrality scores for the onstage matrices
plot_scores(closeness_centrality_onstage, 'Closeness Centrality Scores for Onstage Matrices', 'Closeness Centrality', 'onstage_closeness_centrality', plot_type='bar', sort='year')

# Plot the eigenvector centrality scores for the onstage matrices
plot_scores(eigenvector_centrality_onstage, 'Eigenvector Centrality Scores for Onstage Matrices', 'Eigenvector Centrality', 'onstage_eigenvector_centrality', plot_type='bar', sort='year')

# Plot the degree centralization scores for the exchange matrices
plot_scores(degree_centralization_exchange, 'Degree Centralization Scores for Exchange Matrices', 'Degree Centralization', 'exchange_degree_centralization', plot_type='bar', sort='year')

# Plot the betweenness centrality scores for the exchange matrices
plot_scores(betweenness_centrality_exchange, 'Betweenness Centrality Scores for Exchange Matrices', 'Betweenness Centrality', 'exchange_betweenness_centrality', plot_type='bar', sort='year')

# Plot the closeness centrality scores for the exchange matrices
plot_scores(closeness_centrality_exchange, 'Closeness Centrality Scores for Exchange Matrices', 'Closeness Centrality', 'exchange_closeness_centrality', plot_type='bar', sort='year')

# Plot the eigenvector centrality scores for the exchange matrices
plot_scores(eigenvector_centrality_exchange, 'Eigenvector Centrality Scores for Exchange Matrices', 'Eigenvector Centrality', 'exchange_eigenvector_centrality', plot_type='bar', sort='year')

    