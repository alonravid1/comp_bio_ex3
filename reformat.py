with open('results0.csv', 'r') as res_file:
    with open('results0_reformatted.csv', 'w') as res_file2:
        lines = res_file.readlines()
        res_file2.write("NETWORK_STRUCTURE,POPULATION_SIZE," +
                        "MAX_GENERATIONS,REPLICATION_RATE,MUTATION_RATE,LEARNING_RATE," +
                        "TOURNAMENT_SIZE,BEST_SCORE\n")
        
        for i in range(1, len(lines), 2):
            # get everything between the square brackets in the first part of the line
            arr = lines[i].split("[")[1].split("]")[0].split(",")
            # find the last square bracket in the line
            last_bracket = lines[i].rfind("]")
            params = lines[i][last_bracket+1:].strip("\n")
            score = lines[i+1].strip("\n")
            score = score.strip("best score:")
            res_file2.write("".join(arr) + params + "," + score + "\n")
