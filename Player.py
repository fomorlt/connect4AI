from random import random
import numpy as np

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.depth = 4



    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        abmove = 0 #col of move
        colcounter = 0

        max_player = True

        def game_won(board, player_num):
            player_win_str = '{0}{0}{0}{0}'.format(player_num)
            to_str = lambda a: ''.join(a.astype(str))

            def check_horizontal(b):
                for row in b:
                    if player_win_str in to_str(row):
                        return True
                return False

            def check_verticle(b):
                return check_horizontal(b.T)

            def check_diagonal(b):
                for op in [None, np.fliplr]:
                    op_board = op(b) if op else b
                    
                    root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                    if player_win_str in to_str(root_diag):
                        return True

                    for i in range(1, b.shape[1]-3):
                        for offset in [i, -i]:
                            diag = np.diagonal(op_board, offset=offset)
                            diag = to_str(diag.astype(np.int))
                            if player_win_str in diag:
                                return True
                return False

            return (check_horizontal(board) or
                    check_verticle(board) or
                    check_diagonal(board))

        def terminal_state(board):
            return game_won(board, 1) or game_won(board, 2)




        # Based on pseudocode from Wikipedia and Keith Galli
        # returns max value of a board
        # should return a tuple with a value and a column number

        # obtain max heuristic, now need to make the move that matches that heuristic
        # function is called minimax but actually implements alpha beta pruning
        def minimax(self, board, depth, alpha, beta, max_player):
            #base case
            print("depth: " + str(depth))
            succs = self.compute_successors(board[0], max_player)
            #if depth == 0:
            #    print('---SUCS-----')
            #    print(succs)
            if depth == self.depth or terminal_state(board[0]):
                #always has to return tuple of (boardstate, column value)
                return (self.evaluation_function(board[0]), board[1])
            #recursive
            if max_player:
                value = (-99999999999, -1)
                for boardstatecol in succs:
                    new_value = (minimax(self, boardstatecol, depth + 1, alpha, beta, False)[0], boardstatecol[1])
                    value = new_value if value[0] < new_value[0] else value
                    if value[0] >= beta:
                        break
                    alpha = max(alpha, value[0])
                return value

            else: #minimizing
                value = (99999999999, -1)
                for boardstatecol in succs:
                    new_value = (minimax(self, boardstatecol, depth + 1, alpha, beta, True)[0], boardstatecol[1])
                    value = new_value if value[0] > new_value[0] else value                    
                    if value[0] <= alpha:
                        break
                    beta = min(beta, value[0])
                return value

        wanted_board = minimax(self, (board, 0), 0, -99999999999, 99999999999, max_player)
        #print("-----")
        #print("Wanted board col val:")
        #print(wanted_board[1])
        return wanted_board[1]

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        max_player = True
        def game_won(board, player_num):
            player_win_str = '{0}{0}{0}{0}'.format(player_num)
            to_str = lambda a: ''.join(a.astype(str))

            def check_horizontal(b):
                for row in b:
                    if player_win_str in to_str(row):
                        return True
                return False

            def check_verticle(b):
                return check_horizontal(b.T)

            def check_diagonal(b):
                for op in [None, np.fliplr]:
                    op_board = op(b) if op else b
                    
                    root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                    if player_win_str in to_str(root_diag):
                        return True

                    for i in range(1, b.shape[1]-3):
                        for offset in [i, -i]:
                            diag = np.diagonal(op_board, offset=offset)
                            diag = to_str(diag.astype(np.int))
                            if player_win_str in diag:
                                return True
                return False

            return (check_horizontal(board) or
                    check_verticle(board) or
                    check_diagonal(board))

        def terminal_state(board):
            return game_won(board, 1) or game_won(board, 2)
        

        def expectimax(self, boardcoltuple, depth, is_max):
            succs = self.compute_successors(boardcoltuple[0], is_max)
            #base case
            if depth == self.depth or terminal_state(boardcoltuple[0]):
                emeval = self.evaluation_function(boardcoltuple[0])
                return (emeval, boardcoltuple[1])
            #recursion
            if is_max: # max node
                a = (-999999999999999, -1)
                for succboard in succs:
                    #succboard[0] is board state
                    new_a = (expectimax(self, succboard, depth + 1, False)[0], succboard[1])
                    a = new_a if new_a[0] > a[0] else a
                return a

            else: # chance node, take average of outcomes
                a = (0, -1)
                for succboard in succs:
                    emholder = expectimax(self, succboard, depth + 1, True)
                    a = ((a[0] + ((1 / len(succs)) * emholder[0])), succboard[1])
            return a
        #colmove should be a tuple with values (utility, column)
        colmove = expectimax(self, (board, 0), 0, max_player)
        return colmove[1]

        #raise NotImplementedError('Whoops I don\'t know what to do')

    # Successors
    def compute_successors(self, board, player_bool):
        #Generate successor numpy arrays, append them to the end of a list and return a list of tuples (boardstate, move)
        succ = []
        playertoken = self.player_number
        for col in range(board.shape[1]):
            for row in range(board.shape[0]):
                    boardcopy = board.copy()
                    if not player_bool:
                        #swap token num
                        playertoken = 1 if playertoken == 2 else 1
                        # abs(playertoken - 2) + 1 heheheha
                    if boardcopy[5 - row, col] == 0:
                        boardcopy[5 - row, col] = playertoken
                        succ.append((boardcopy, col))
                        #tuple with board state and the column value the move was made in
                        break
                    #only break once spot found in a row, if not then keep climbing to
                    #higher row until spot available.
        return succ
            

        

    # Evaluation Function
    # Returns a value of a board state
    # Start by reading board - check for various cases and return value 
    # to be used in alpha-beta 

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        # evaluates utility of given state of the board.

        # instead of just returning a single value, having the score add on for more pieces next to
        # each other. account for blocking by rewarding boards with open spaces around
        # making various different states:

       # Get opposing player id
        enemy_number = abs(self.player_number - 2) + 1
      
        #kernels
        kernel_scores_horizontal = [
          ('000{0}'.format(self.player_number), 250), 
          ('00{0}0'.format(self.player_number), 250), 
          ('0{0}00'.format(self.player_number), 250), 
          ('{0}000'.format(self.player_number), 250), 
          ('00{0}{0}'.format(self.player_number), 5000), 
          ('0{0}{0}0'.format(self.player_number), 5000), 
          ('{0}{0}00'.format(self.player_number), 5000), 
          ('0{0}{0}{0}'.format(self.player_number), 100000), 
          ('{0}{0}{0}0'.format(self.player_number), 100000), 
          ('{0}{0}{0}{0}'.format(self.player_number), 2000000), 
          ('000{0}'.format(enemy_number), -500), 
          ('00{0}0'.format(enemy_number), -500), 
          ('0{0}00'.format(enemy_number), -500), 
          ('{0}000'.format(enemy_number), -500), 
          ('00{0}{0}'.format(enemy_number), -10000), 
          ('0{0}{0}0'.format(enemy_number), -10000), 
          ('{0}{0}00'.format(enemy_number), -10000), 
          ('0{0}{0}{0}'.format(enemy_number), -20000), 
          ('{0}{0}{0}0'.format(enemy_number), -20000), 
          ('{0}{0}{0}{0}'.format(enemy_number), -4000000)
        ]

        kernel_scores_vertical = [
          ('000{0}'.format(self.player_number), 250), 
          ('00{0}{0}'.format(self.player_number), 5000), 
          ('0{0}{0}{0}'.format(self.player_number), 100000),  
          ('{0}{0}{0}{0}'.format(self.player_number), 2000000), 
          ('000{0}'.format(enemy_number), -500), 
          ('00{0}{0}'.format(enemy_number), -10000), 
          ('0{0}{0}{0}'.format(enemy_number), -200000),  
          ('{0}{0}{0}{0}'.format(enemy_number), -4000000)
        ]
        #board score
        board_util = 0
        
        #takes parameter a, converts it into a string of numbers for reading
        to_str = lambda a: ''.join(a.astype(str))

        # calculate horizontal "goodness"
        #checks, takes board b in.
        #need to add for each check to see if 
        for row in board:
            rowstr = to_str(row)
            for scorestr in kernel_scores_horizontal:
                if scorestr[0] in rowstr:
                    board_util += scorestr[1]


        # calculate vertical "goodness"
        boardcopytranspose = board.T
        for col in boardcopytranspose:
            colstr = to_str(col)
            for scorestr in kernel_scores_vertical:
                if scorestr[0] in colstr:
                    board_util += scorestr[1]

        
        # calculate diagonal "goodness"
        #def diagonal_eval(b):
        for op in [None, np.fliplr]:
            op_board = op(board) if op else board
            
            root_diag = np.diagonal(op_board, offset=0).astype(np.int)
            # string of diagonals has been acquired, now time to check
            diagstr = to_str(root_diag)
            for scorestr in kernel_scores_horizontal:
                if scorestr[0] in diagstr:
                    board_util += scorestr[1]
            
            for i in range(1, board.shape[1]-3):
                for offset in [i, -i]:
                    diag = np.diagonal(op_board, offset=offset)
                    diag = to_str(diag.astype(np.int))
                    for scorestr in kernel_scores_horizontal:
                        if scorestr[0] in diag:
                            board_util += scorestr[1]
        

        # start building utility value of board, adding onto board_util for every
        # piece with other pieces nearby
        #print("----------board util calculated----------")
        #print(board_util)
        return board_util
    


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        #print(board)
        valid_cols = []
        #making a move
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)
        #print(valid_cols)
        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

