from converter.pgn_data import PGNData
import pandas as pd
import ast

# pgn_data = PGNData("lichess_db_standard_rated_2013-01.pgn")
# result = pgn_data.export()
# if result.is_complete:
#     # read the games file
#     games_df = result.get_games_df()
#     print(games_df.head())

# combined_df = result.get_combined_df()
# games = pd.read_csv('lichess_db_standard_rated_2013-01_game_info.csv')
# moves = pd.read_csv('lichess_db_standard_rated_2013-01_moves.csv')
# data = pd.merge(games, moves, on = 'game_id')
# data.to_csv('full_chess_data.csv')

# df = pd.read_csv('full_chess_data.csv')

# #dropping unneeded columns
# df = df.drop(columns = ['white_title', 'black_title', 'site', 'date_played', 'round', 'white', 'black', 'white_rating_diff', 'black_rating_diff', 'winner', 'winner_elo', 'loser', 'loser_elo', 'winner_loser_elo_diff', 'utc_date', 'utc_time', 'variant', 'date_created', 'ply_count', 'file_name'])

# #dropping games that are not "rated classical" -- This includes blitz and bullet games
# df = df.drop(df[df['event'] != 'Rated Classical game'].index)

# #Reloading df in order decide on new drops
# df.to_csv('first_filter.csv')

# data = pd.read_csv('first_filter.csv')

# #dropping more columns
# data = data.drop(columns = ['Unnamed: 0.1'])
# data = data.drop(data[data['is_game_over'] == 0].index)

# data.to_csv('second_filter.csv')

# data = pd.read_csv('second_filter.csv')
# data = data.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1', 'event', 'move', 'notation','from_square', 'to_square', 'piece', 'color', 'fen', 'white_pawn_count', 'black_pawn_count', 'white_queen_count', 'black_queen_count', 'white_bishop_count', 'black_bishop_count', 'white_knight_count', 'black_knight_count', 'white_rook_count', 'black_rook_count'])
# data = data.drop(columns = ['fen_row1_white_count','fen_row2_white_count','fen_row3_white_count','fen_row4_white_count','fen_row5_white_count','fen_row6_white_count','fen_row7_white_count','fen_row8_white_count','fen_row1_white_value','fen_row2_white_value','fen_row3_white_value','fen_row4_white_value','fen_row5_white_value','fen_row6_white_value','fen_row7_white_value','fen_row8_white_value','fen_row1_black_count','fen_row2_black_count','fen_row3_black_count','fen_row4_black_count','fen_row5_black_count','fen_row6_black_count','fen_row7_black_count','fen_row8_black_count','fen_row1_black_value','fen_row2_black_value','fen_row3_black_value','fen_row4_black_value','fen_row5_black_value','fen_row6_black_value','fen_row7_black_value','fen_row8_black_value'])
# data = data.drop(data[data['white_elo'] == '?'].index)
# data = data.drop(data[data['black_elo'] == '?'].index)
# data = data.drop(data[data['move_no'] < 7].index)
# data.to_csv('filtered.csv')
#Convert result column into 1, 0, or -1 based on result
def convertRes(result):
    if result == '1-0':
        return 1
    elif result == '0-1':
        return -1
    else: #case when there is a draw
        return 0

#Convert ECO into decimal numbers
#ECO's come in the form "lETTER NUMBER NUMBER" so we will just map the letter to a number and return the eco as an integer (ie. A00 = 000 = 0)
def convertECO(eco):
    mappings = {'A': '0', 'B': '1', 'C': '2', 'D': '3', 'E': '4'}
    res = mappings.get(eco[0]) + eco[1:]
    return int(res)

#Converting moves into mapped vector
#Move encoding plan:
#1. split into individual moves, and remove move number
#2. create a vector for each move with following encoding: [check, checkmate, capture, queenside castle, kingside castle, King, Queen, Rook, Bishop, Knight, Pawn]
#check denoted with "+", checkmate denoted with "#", capture denoted with "X", queenside castle is "0-0-0", kingside is "0-0", in Opening is binary "if we are still in opening or not", promotion denoted with "="
def convertMove(move):
    #if the move has already been converted
    if type(move) != str:
        return '~'

    # row = 0
    # col = 0
    
    king = 0
    queen = 0
    rook = 0
    bishop = 0
    knight = 0
    pawn = 0
    
    capture = 0
    promo = 0
    castle_ks = 0
    castle_qs = 0
    check = 0
    checkmate = 0
    
    pieces = ['K', 'Q', 'R', 'B', 'N']
    move_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e':5, 'f':6, 'g':7, 'h':8}
    
    
    if '0-0-0' in move:
        castle_qs = 1
        king = 1
        rook = 1
    elif '0-0' in move:
        castle_ks = 1
        king = 1
        rook = 1
    else:
        # fmove = move.replace('#','').replace('=','').replace('+', '').replace('x','')
        # fmove = fmove[len(fmove)-2:]
        # row = move_map.get(fmove[0])
        # col = fmove[1]
        if 'K' in move:
            king = 1
        elif 'Q' in move:
            queen = 1
        elif 'R' in move:
            rook = 1
        elif 'B' in move:
            bishop = 1
        elif 'N' in move:
            knight = 1
        elif move[0] not in pieces:
            pawn = 1
        
        
        if 'x' in move:
            check = 1
        if '=' in move:
            promo = 1
        if '+' in move:
            check = 1
        elif '#' in move:
            checkmate = 1
    
 
    return checkmate, check, promo, capture, castle_qs, castle_ks, queen, king, rook, bishop, knight, pawn

data = pd.read_csv('filtered.csv')
data = data.drop(data[data['move_no'] > 150].index)

#Making an average rating column
data['avg_rating'] = data.apply(lambda row: (row.white_elo + row.black_elo)//2, axis = 1)

#Splitting moves and padding columns
splitMoves = data['move_sequence'].str.split(pat = '|', expand = True)
numCols = splitMoves.shape[1]
splitMoves.columns = ['move' + str(i) for i in range(1,numCols+1)]
cols = splitMoves.columns.tolist()
for r in range(len(splitMoves.index)):
    for c in range(len(splitMoves.columns)):
        move = splitMoves.iat[r,c]
        if not move: #converting null values into a tuple of 0's
            splitMoves.iat[r,c] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        else:
            converted = convertMove(move)
            if converted != '~':
                splitMoves.iat[r,c] = converted

#Flattening the move tuples into separate rows for the dataset
for i in range(1, 151):
    splitMoves = pd.concat([splitMoves, splitMoves[f'move{i}'].apply(lambda x: pd.Series(x, index=[f'move_{i}_checkmate', f'move_{i}_check', f'move_{i}_promo', f'move_{i}_capture', f'move_{i}_castle_qs', f'move_{i}_castle_ks', f'move_{i}_queen', f'move_{i}_king', f'move_{i}_rook', f'move_{i}_bishop', f'move_{i}_knight', f'move_{i}_pawn']))], axis=1)
    # move_i_values = pd.DataFrame(splitMoves[f'move{i}'].tolist(), columns=[f'move_{i}_checkmate', f'move_{i}_check', f'move_{i}_promo', f'move_{i}_capture', f'move_{i}_castle_qs', f'move_{i}_castle_ks', f'move_{i}_queen', f'move_{i}_king', f'move_{i}_rook', f'move_{i}_bishop', f'move_{i}_knight', f'move_{i}_pawn'])
    # splitMoves = pd.concat([splitMoves, move_i_values], axis = 1)
    splitMoves = splitMoves.drop(columns =  [f'move{i}'])

#Recombine dataframes
data = pd.concat([data, splitMoves], axis = 1)
data = data.drop(columns = ['termination', 'move_sequence', 'player', 'notation', 'time_control', 'move_no_pair', 'white_elo', 'black_elo'])
print(data['eco'].unique())

#Converting ECO's into integers
for i in range(len(data['eco'])):
    value = data.iat[i, data.columns.get_loc('eco')]
    data.iat[i, data.columns.get_loc('eco')] = convertECO(value)



#Converting res into 1, -1, 0
for i in range(len(data['result'])):
    value = data.iat[i, data.columns.get_loc('result')]
    data.iat[i, data.columns.get_loc('result')] = convertRes(value)


data.to_csv('final_data.csv')
