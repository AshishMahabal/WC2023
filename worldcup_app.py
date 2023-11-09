import streamlit as st

from worldcup_utils import (load_data, plot_matrix_chart, generate_match_results,
                            order_teams_by_points_and_nrr, get_index_for_preselection,
                            calculate_points, calculate_nrr, country_winloss,
                            generate_results_matrix, generate_standings_and_plot,
                            get_country_details,plot_country_barchart,
                            get_country_data, plot_country_graph, 
                            get_victory_margin,add_line, plot_decagon,
                            ground_stats, toss_decision_outcome)

def main(df, countries):
    st.title('World Cup 2023 Results')
    st.write('Choose options to see desired details')
    
    choice = st.radio(
        "Choose a view:",
        ("Overall Stats", "Per Country Analysis")
    )
    # # Create two columns
    #col101, col102 = st.columns(2)

    # # Place a radio button in each column
    # with col101:
    #     overall = st.radio("Choose a view:", ["Overall Stats"], key="1")
    # with col102:
    #     per_country = st.radio("", ["Per Country Analysis"], key="2")

    # # Determine the selected choice
    # if overall:
    #     choice = "Overall Stats"
    # elif per_country:
    #     choice = "Per Country Analysis"

    # # Use buttons to emulate radio selection and store choice in a session state
    # if not hasattr(st.session_state, 'radio_choice'):
    #     st.session_state.radio_choice = None

    # if col101.button("Overall Stats"):
    #     st.session_state.radio_choice = "Overall Stats"
    # elif col102.button("Per Country Analysis"):
    #     st.session_state.radio_choice = "Per Country Analysis"

    # choice = st.session_state.radio_choice

    if choice == "Overall Stats":
        # ... display overall statistics ...
        st.write("Overall statistics go here.")

        # Create a row with three columns
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        # Place a checkbox in each column
        with col1:
            checkbox1 = st.checkbox("Standings and Victories", value=True)
        with col2:
            checkbox2 = st.checkbox("Standings (sorted)", value=True)
        with col3:
            checkbox3 = st.checkbox("Head to Head", value=True)
        with col4:
            checkbox4 = st.checkbox("Per match details")
        with col5:
            checkbox5 = st.checkbox("Based on Toss")
        with col6:
            checkbox6 = st.checkbox("Per ground details")

        if checkbox1:
            st.write('Who beat whom')
            plot_decagon(df)

        if checkbox2:
            st.write('Team standings')
            generate_standings_and_plot(df)

        if checkbox3:
            st.write('Head to Head')
            #st.write('Pak/NZ NRR off by 0.2 - need to incorporate DL')
            plot_matrix_chart(results_matrix, sorted_countries, team_points, team_nrr, df)

        # Show Raw Data
        if checkbox4:
            st.write('Per match details')
            st.write(df)

        if checkbox5:
            st.write('Based on Toss')
            toss_decision_outcome(df)

        if checkbox6:
            st.write('Per ground stats')
            ground_stats(df)


    elif choice == "Per Country Analysis":
        # Create a row with columns
        col11, col12, col13, col14 = st.columns(4)
        # Place a checkbox in each column
        with col11:
            checkbox11 = st.checkbox("Match details")
        with col12:
            checkbox12 = st.checkbox("Difference", value=True)
        with col13:
            checkbox13 = st.checkbox("Tree", value=True)
        with col14:
            checkbox14 = st.checkbox("Bat, Win, Toss ...", value=True)
        # Dropdown for country selection
        index_to_preselect = get_index_for_preselection(countries, 'Ind')
        selected_country = st.selectbox('Choose a country:', countries,index=index_to_preselect)
        country_specific_df = get_country_data(df,selected_country)
        if checkbox11:
            st.write('All the rows')
            st.write(country_specific_df)
        if checkbox12:
            st.write('How were they beaten')
            plot_country_barchart(df, selected_country)
        if checkbox13:
            st.write('Who beat whom')
            plot_country_graph(selected_country, country_specific_df, countries)
        if checkbox14:
            st.write('Bat first, toss etc. stats')
            country_winloss(df, selected_country)
        
    # Compute matrix
#    team_points, team_nrr = compute_team_metrics(data, countries)
    # Get points and NRR for sorting
    #team_points = calculate_points(df, countries)
    #team_nrr = calculate_nrr(df, countries)
    #sorted_countries = sorted(countries, key=lambda x: (team_points[x], team_nrr[x]), reverse=True)
#    matrix = create_match_matrix(data, sorted_countries)
#    results_matrix = generate_results_matrix(df, sorted_countries)
    
    # Plot matrix chart
    #plot_matrix_chart(results_matrix, sorted_countries, team_points, team_nrr, df)
    
    # Any other visualizations or features can be added here...

    # Divider
    st.write("---")

    # Source Code
    st.write("**Source Code:** [GitHub Repo](https://github.com/AshishMahabal/WC2023) PRs welcome.")

    # Version
    st.write("**Version:** 0.1")

    # Disclaimer
    st.write("### Disclaimer")
    st.write("This app makes no guarantees - use at your own peril.")

# Run Streamlit app
if __name__ == '__main__':
    filename = "worldcup2023results.csv"
    df = load_data(filename)
#    generate_standings_and_plot(df)
#    data = load_data()
    countries = ["Ind", "Aus", "NZ", "SL", "Ban", "Pak", "Eng", "SA", "Ned", "Afg"]
    team_points = calculate_points(df, countries)
    team_nrr = calculate_nrr(df, countries)
    sorted_countries = sorted(countries, key=lambda x: (team_points[x], team_nrr[x]), reverse=True)
    results_matrix = generate_results_matrix(df, sorted_countries)
    main(df, countries)

