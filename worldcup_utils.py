# worldcup_utils.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import networkx as nx

# ... All your previously discussed functions go here ...

def load_data(filename):
    return pd.read_csv(filename)

def calculate_points(df, countries):
    points = {country: 0 for country in countries}
    for index, row in df.iterrows():
        result = row['Result']
        bat1 = row['Bat1']
        bat2 = row['Bat2']

        if result == 'draw' or result == 'tie':
            points[bat1] += 1
            points[bat2] += 1
        else:
            points[result] += 2
    return points

def convert_overs(overs):
    # Convert fractional overs e.g., 35.3 (35 overs and 3 balls) to 35.5
    whole_overs = int(overs)
    balls = overs - whole_overs
    return whole_overs + balls * 10 / 6

def calculate_nrr(df, countries):
    # Initialize dictionaries to accumulate values
    total_runs_scored = {}
    total_overs_batted = {}
    total_runs_conceded = {}
    total_overs_bowled = {}

    for index, row in df.iterrows():
        bat1 = row['Bat1']
        bat2 = row['Bat2']

        runs_scored1 = row['Runs1']
        overs_batted1 = convert_overs(row['Overs1'] if row['wickets1'] < 10 else row['maxovers1'])
        
        runs_scored2 = row['Runs2']
        overs_batted2 = convert_overs(row['Overs2'] if row['wickets2'] < 10 else row['maxovers2'])

        # Update the total runs scored and overs batted for both teams
        total_runs_scored[bat1] = total_runs_scored.get(bat1, 0) + runs_scored1
        total_overs_batted[bat1] = total_overs_batted.get(bat1, 0) + overs_batted1

        total_runs_scored[bat2] = total_runs_scored.get(bat2, 0) + runs_scored2
        total_overs_batted[bat2] = total_overs_batted.get(bat2, 0) + overs_batted2

        # Update the total runs conceded and overs bowled for both teams
        total_runs_conceded[bat1] = total_runs_conceded.get(bat1, 0) + runs_scored2
        total_overs_bowled[bat1] = total_overs_bowled.get(bat1, 0) + overs_batted2

        total_runs_conceded[bat2] = total_runs_conceded.get(bat2, 0) + runs_scored1
        total_overs_bowled[bat2] = total_overs_bowled.get(bat2, 0) + overs_batted1

    # Calculate the NRR for each team
    nrr_dict = {}
    for team in total_runs_scored.keys():
        nrr = (total_runs_scored[team] / total_overs_batted[team]) - (total_runs_conceded[team] / total_overs_bowled[team])
        #st.write(team,total_runs_scored[team],total_overs_batted[team],total_runs_conceded[team],total_overs_bowled[team])
        nrr_dict[team] = nrr
        
    return nrr_dict

def order_teams_by_points_and_nrr(filename, countries):
    df = load_data(filename)
    points = calculate_points(df, countries)
    nrr = calculate_nrr(df)

    # Combine points and NRR into a single DataFrame for sorting
    combined = pd.DataFrame(list(points.items()), columns=['Team', 'Points'])
    combined['NRR'] = combined['Team'].map(nrr)
    
    # Sort by points first, then NRR
    combined = combined.sort_values(by=['Points', 'NRR'], ascending=[False, False])
    
    return combined

def get_matches_played(df):
    matches_played = {}
    for team in set(df["Bat1"].tolist() + df["Bat2"].tolist()):
        matches_played[team] = len(df[(df["Bat1"] == team) | (df["Bat2"] == team)])
    return matches_played

def plot_bubble_chart(teams, points, nrr, matches_played):
    # Set the figure size and style
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    #st.write(nrr)
    print(matches_played)

    # Create a custom colormap for NRR - bluer for positive and redder for negative
    cmap = plt.cm.get_cmap('coolwarm_r')

    # Create the scatter plot
    plt.scatter(teams, points, s=[mp*100 for mp in matches_played], c=nrr, cmap=cmap, edgecolors='black', alpha=0.7)

    # Create a colorbar
    plt.colorbar().set_label('Net Run Rate (NRR)')

    # Add titles and labels
    plt.title("World Cup 2023 Standings", fontsize=16)
    plt.xlabel("Teams", fontsize=14)
    plt.ylabel("Points", fontsize=14)

    # Adjust Y-axis and rotate X-axis labels for better readability
    plt.ylim(min(points) - 1, max(points) + 1)
    plt.xticks(rotation=45)
    
    # Display bubble size significance
    for i, team in enumerate(teams):
        plt.annotate(matches_played[i], (team, points[i]), fontsize=10, ha='center', va='center')

    # Display the plot
    plt.tight_layout()
    st.pyplot(plt.gcf()) 
#    plt.show()

def generate_standings_and_plot(df):
    countries = ["Aus", "Eng", "SA", "NZ", "Pak", "Ind", "SL", "Ban", "Afg", "Ned"]

    # Get the points, NRR, and matches played for each country
    points_dict = calculate_points(df, countries)
    nrr_dict = calculate_nrr(df, countries)
    print(nrr_dict)
    matches_played_dict = get_matches_played(df)
    print(matches_played_dict)

    # Prepare a list of tuples for sorting
    team_data = [(country, points_dict.get(country, 0), nrr_dict.get(country, 0), matches_played_dict.get(country, 0)) for country in countries]
    
    # Sort based on points and then NRR
    sorted_team_data = sorted(team_data, key=lambda x: (x[1], x[2]), reverse=True)
#    st.write(sorted_team_data)

    # Extract the sorted values
    countries_sorted = [data[0] for data in sorted_team_data]
    points_sorted = [data[1] for data in sorted_team_data]
    nrr_sorted = [data[2] for data in sorted_team_data]
    matches_played_sorted = [data[3] for data in sorted_team_data]
    print(nrr_sorted)

    # Call the plotting function
    plot_bubble_chart(countries_sorted, points_sorted, nrr_sorted, matches_played_sorted)

def generate_results_matrix(df, countries):
    matrix = np.full((len(countries), len(countries)), np.nan)  # Initialize matrix with NaNs (for unplayed matches)
    
    for i, team1 in enumerate(countries):
        for j, team2 in enumerate(countries):
            if team1 == team2:
                matrix[i, j] = -3  # Special value for the diagonal; we'll replace later
                continue
            
            match = df[((df["Bat1"] == team1) & (df["Bat2"] == team2)) | 
                       ((df["Bat1"] == team2) & (df["Bat2"] == team1))]
            
            if not len(match):
                continue

            if match["Result"].values[0] == "draw" or match["Result"].values[0] == "tie":
                matrix[i, j] = 1
            elif match["Result"].values[0] == team1:
                matrix[i, j] = 2
            else:
                matrix[i, j] = 0

    return matrix

def get_victory_margin(data, team1, team2):
    match_data = data[((data['Bat1'] == team1) & (data['Bat2'] == team2)) | 
                      ((data['Bat1'] == team2) & (data['Bat2'] == team1))]
    
    if not len(match_data):
        return None
    
    match_data = match_data.iloc[0]  # Assume only one match for now between the two teams
    
    if match_data['Result'] == team1:
        if match_data['Bat1'] == team1:
            return f"by {match_data['Target'] - match_data['Runs2']} runs"
        else:
            return f"by {10 - match_data['wickets2']} wickets"
    elif match_data['Result'] == team2:
        if match_data['Bat1'] == team2:
            return f"by {match_data['Target'] - match_data['Runs2']} runs"
        else:
            return f"by {10 - match_data['wickets2']} wickets"
    else:
        return None

def plot_matrix_chart(matrix, countries, team_points, nrr, data):
    plt.figure(figsize=(12, 12))
    
    # Define custom colormap for off-diagonal elements
    colors = [(0.8, 0.8, 0.8), (1, 0, 0), (1, 1, 0), (0, 0, 1)]  # gray -> red -> yellow -> blue
    cm = LinearSegmentedColormap.from_list("custom_div_cmap", colors, N=4)
    
    # Plot heatmap with modified colormap
    ax = sns.heatmap(matrix, cmap=cm, cbar=False, xticklabels=countries, yticklabels=countries, 
                 linewidths=.5, linecolor='white', annot=False)

    # Move x-labels to the top
    ax.xaxis.tick_top() 
    
    # Custom annotations and box resizing
    for i in range(len(countries)):
        for j in range(len(countries)):
            #value = matrix[i, j]
            margin = get_victory_margin(data, countries[i], countries[j])
            
            # Diagonal cells
            if i == j:
                color = "white"  # Default text color
                if i < 4:
                    cell_color = "blue"
                    color = "white"
                elif i < 7:
                    cell_color = "yellow"
                    color = "black"  # Adjust text color for better visibility on yellow
                else:
                    cell_color = "red"
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=cell_color))
                ax.text(j+0.5, i+0.5, f"{int(team_points[countries[i]])} ({nrr[countries[i]]:.2f})", 
                       horizontalalignment='center', verticalalignment='center', color=color)

            # Off-diagonal cells
            else:
                text = ""
                if not np.isnan(team_points[countries[i]]):
                    # ... (rest of the color adjustments as before)
                    #text = f"{int(team_points[countries[i]])}"
                    if margin:
                        text += f"\n({margin})"
                ax.text(j+0.5, i+0.5, text, horizontalalignment='center', 
                        verticalalignment='center', color='black')

    plt.title("Head-to-Head Results (%d matches)" % len(data), fontsize=16)
    st.pyplot(plt.gcf()) 
#    plt.show()

# Function to add a line for a match result
def add_line(result, ax, countries, angles):
    tokens = result.split()
    winner = tokens[0]
    loser = tokens[2]

    winner_index = countries.index(winner)
    loser_index = countries.index(loser)

    winner_x = np.cos(angles[winner_index])
    winner_y = np.sin(angles[winner_index])
    loser_x = np.cos(angles[loser_index])
    loser_y = np.sin(angles[loser_index])

    # Calculate the midpoint
    midpoint_x = (winner_x + loser_x) / 2
    midpoint_y = (winner_y + loser_y) / 2

    # Determine line colors based on the match result
    if "tied" in result or "drew" in result:
        line_color = 'yellow'
    else:
        line_color = 'blue' if winner == countries[winner_index] else 'red'

    ax.plot([winner_x, midpoint_x], [winner_y, midpoint_y], color=line_color, linewidth=2)
    ax.plot([midpoint_x, loser_x], [midpoint_y, loser_y], color='red', linewidth=2)
    #print(result, winner_x, midpoint_x, loser_x, winner_y, midpoint_y, loser_y)
    print(winner, countries[winner_index])

    # Add arrow at the midpoint
    if line_color == 'blue':
        arrowprops = dict(arrowstyle='<|-', color='blue', lw=2)
        ax.annotate('', xy=(midpoint_x, midpoint_y), xytext=(loser_x, loser_y), arrowprops=arrowprops)

def plot_decagon(df):
    countries = ["Aus", "Eng", "SA", "NZ", "Pak", "Ind", "SL", "Ban", "Afg", "Ned"]
    countries = ["Ind", "Aus", "NZ", "SL", "Ban", "Pak", "Eng", "SA", "Ned", "Afg"]
    
    # Get the points, NRR, and matches played for each country
    points_dict = calculate_points(df, countries)
    nrr_dict = calculate_nrr(df, countries)
    print(nrr_dict)
    matches_played_dict = get_matches_played(df)
    print(matches_played_dict)
    
    # Prepare a list of tuples for sorting
    team_data = [(country, points_dict.get(country, 0), nrr_dict.get(country, 0), matches_played_dict.get(country, 0)) for country in countries]
    
    # Sort based on points and then NRR 
    sorted_team_data = sorted(team_data, key=lambda x: (x[1], x[2]), reverse=True)
    
    # Extract the sorted values
    sorted_countries = [data[0] for data in sorted_team_data]
    points_sorted = [data[1] for data in sorted_team_data]
    nrr_sorted = [data[2] for data in sorted_team_data]
    matches_played_sorted = [data[3] for data in sorted_team_data]
    print(nrr_sorted)

    # Split countries into top, middle, and bottom based on points
    top_countries = sorted_countries[:4]
    middle_countries = sorted_countries[4:7]
    bottom_countries = sorted_countries[7:]

    # Create a 10-sided polygon to represent countries
    num_countries = len(countries)
    angle_step = 2 * np.pi / num_countries
    angles = [i * angle_step for i in range(num_countries)]

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Add lines for each match result
    for result in generate_match_results(df):
        add_line(result, ax, countries, angles)

    # Plot colored blobs at the location of each country and print country names below
    for country, angle in zip(countries, angles):
        x = 1.2 * np.cos(angle)
        y = 1.2 * np.sin(angle)
        country_color = 'blue' if country in top_countries else ('yellow' if country in middle_countries else 'red')
#        country_size = country_points[country] * 60  # Adjust the multiplier for appropriate blob size
        country_size = points_dict[country] * 60  # Adjust the multiplier for appropriate blob size
        ax.scatter(x, y, s=country_size, c=country_color, edgecolor='black', linewidth=1)
        ax.text(x, y - 0.3, country, ha='center', va='center', fontsize=10)

    # Customize plot settings
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("World Cup 2023 (after %d matches)" % len(df))
    #plt.savefig('20231020.png')

    # Show the plot
    st.pyplot(plt.gcf()) 
#    plt.show()

def generate_match_results(df):
    results = []
    
    for index, row in df.iterrows():
        team1 = row['Bat1']
        team2 = row['Bat2']
        result = row['Result']

        if result == 'tie':
            results.append(f"{team1} tied {team2}")
        elif result == 'draw':
            results.append(f"{team1} drew {team2}")
        else:
            if result == team1:
                results.append(f"{team1} beat {team2}")
            else:
                results.append(f"{team2} beat {team1}")
    
    return results

def get_country_data(df, country):
    return df[(df['Bat1'] == country) | (df['Bat2'] == country)]

def plot_country_graph(country, df, countries):
    G = nx.MultiDiGraph()  # MultiDiGraph to allow for multiple edge types (like bidirectional for draws)

    # Filtering matches involving the country.
    relevant_matches = df[(df['Bat1'] == country) | (df['Bat2'] == country)]

    for _, row in relevant_matches.iterrows():
        if row['Result'] == country:  # If the country won the match
            G.add_edge(country, row['Bat2'] if row['Bat1'] == country else row['Bat1'])
        elif row['Result'] in countries:  # If the country lost the match
            G.add_edge(row['Bat2'] if row['Bat1'] == country else row['Bat1'], country)
        elif row['Result'] == "draw" or row['Result'] == "tie":
            # Add bidirectional arrows for draws and ties
            G.add_edge(country, row['Bat2'] if row['Bat1'] == country else row['Bat1'], relation="draw")
            G.add_edge(row['Bat2'] if row['Bat1'] == country else row['Bat1'], country, relation="draw")

    pos = nx.shell_layout(G)

    color_map = []
    for node in G:
        if node == country:
            color_map.append('black')
        elif G.out_degree(node) > 0 and G.in_degree(node) == 0:
            color_map.append('blue')  # Winners
        elif G.in_degree(node) > 0 and G.out_degree(node) == 0:
            color_map.append('red')  # Losers
        else:
            color_map.append('green')  # For bidirectional arrows (draws/ties)

    edge_colors = ['gray' if G[u][v][0].get('relation') != "draw" else 'orange' for u, v in G.edges()]

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color=color_map, font_size=15, width=3, edge_color=edge_colors, arrowsize=20, font_color='white')
    plt.title(f"Matches involving {country}\n")
    
    st.pyplot(plt.gcf())

def get_index_for_preselection(items, item_name):
    try:
        return items.index(item_name)
    except ValueError:
        return 0

#index_to_preselect = get_index_for_preselection(countries, 'Ind')

