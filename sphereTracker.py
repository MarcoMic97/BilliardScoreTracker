import cv2
import numpy as np

# Initialize global variables for scores and players
round_scores = []
total_scores = []
current_team_index = 0
scored = False
players = []
teams = []

def setup_game():
    global round_scores, total_scores, players, teams

    # Input the number of players
    num_players = int(input("Enter the number of players (2 to 4): "))
    
    # Check if they are playing in teams
    team_mode = input("Are players playing in teams? (y/n): ").strip().lower()

    if team_mode == 'y':
        num_teams = int(input(f"Enter the number of teams (2 to {num_players}): "))
        teams = [[] for _ in range(num_teams)]
        
        # Assign players to teams
        for i in range(num_players):
            player_name = input(f"Enter name of player {i+1}: ")
            team_number = int(input(f"Assign {player_name} to team (1 to {num_teams}): ")) - 1
            teams[team_number].append(player_name)
        players = [" & ".join(team) for team in teams]
    else:
        # If no teams, treat each player as their own team
        for i in range(num_players):
            player_name = input(f"Enter name of player {i+1}: ")
            players.append(player_name)

    # Initialize the scores for each player/team
    round_scores = [0] * len(players)
    total_scores = [0] * len(players)

def detect_spheres(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    lower_dark_red = np.array([160, 100, 100])
    upper_dark_red = np.array([180, 255, 255])
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([15, 255, 255])
    
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    dark_red_mask = cv2.inRange(hsv_frame, lower_dark_red, upper_dark_red)
    orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)
    
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dark_red_contours, _ = cv2.findContours(dark_red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_spheres = {'white': [], 'dark_red': [], 'orange': []}
    
    for contour in white_contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 10:
            detected_spheres['white'].append((int(x), int(y), int(radius)))
    
    for contour in dark_red_contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 10:
            detected_spheres['dark_red'].append((int(x), int(y), int(radius)))
    
    for contour in orange_contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 10:
            detected_spheres['orange'].append((int(x), int(y), int(radius)))
    
    return detected_spheres

def track_spheres():
    global round_scores, total_scores, current_team_index, scored
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        detected_spheres = detect_spheres(frame)
        
        for (x, y, radius) in detected_spheres['white']:
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        
        for (x, y, radius) in detected_spheres['dark_red']:
            cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)

        if len(detected_spheres['orange']) == 2 and not scored:
            (x1, y1, r1) = detected_spheres['orange'][0]
            (x2, y2, r2) = detected_spheres['orange'][1]
            
            cv2.circle(frame, (x1, y1), r1, (0, 165, 255), 2)
            cv2.circle(frame, (x2, y2), r2, (0, 165, 255), 2)
            
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            if distance <= r1 + r2:
                round_scores[current_team_index] += 4
                scored = True
                print(f"Round Score for {players[current_team_index]}: {round_scores[current_team_index]}")
        
        # Display the current round score and total score for the current team/player
        cv2.putText(frame, f"Round Score: {round_scores[current_team_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Score: {total_scores[current_team_index]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Current Team: {players[current_team_index]}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Billiard Ball Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            total_scores[current_team_index] += round_scores[current_team_index]
            round_scores[current_team_index] = 0
            scored = False
            print(f"New Round! Total Score for {players[current_team_index]}: {total_scores[current_team_index]}")
            
            # Move to the next team/player
            current_team_index = (current_team_index + 1) % len(players)
            print(f"Next Team: {players[current_team_index]}")
    
    cap.release()
    cv2.destroyAllWindows()

# Setup the game before starting the tracking
setup_game()

# Start tracking the spheres
track_spheres()
