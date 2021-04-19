from typing import List

import matplotlib.pyplot as plt
import networkx as nx

import parameters


class Visualize:

    __graph = nx.Graph()
    __frame_delay = parameters.FRAME_DELAY
    __board_size = parameters.SIZE

    @staticmethod
    def __add_node_to_graph(graph, position):
        graph.add_node(position)

    @staticmethod
    def __add_edge_to_graph(graph, u_position, v_position):
        graph.add_edge(u_position, v_position)

    @staticmethod
    def __get_filled_nodes(board):
        filled_player_1 = []
        filled_player_2 = []
        for index in range(Visualize.__board_size ** 2):
            if board[index] == 1:
                filled_player_1.append((index // Visualize.__board_size, index % Visualize.__board_size))
            if board[index] == 2:
                filled_player_2.append((index // Visualize.__board_size, index % Visualize.__board_size))
        return filled_player_1, filled_player_2

    @staticmethod
    def __get_empty_nodes(board):
        empty_positions = []
        for index in range(Visualize.__board_size ** 2):
            if board[index] == 0:
                empty_positions.append((index // Visualize.__board_size, index % Visualize.__board_size))
        return empty_positions

    @staticmethod
    def __get_legal_positions(board):
        legal_positions = []
        for index in range(Visualize.__board_size ** 2):
            legal_positions.append((index // Visualize.__board_size, index % Visualize.__board_size))
        return legal_positions

    @classmethod
    def initialize_board(cls, state):
        board = state[1:]
        size = Visualize.__board_size
        legal_positions = Visualize.__get_legal_positions(board)
        edges = set([(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)])

        for i in range(size):
            for j in range(size):
                Visualize.__add_node_to_graph(Visualize.__graph, (i, j))

        for x, y in legal_positions:
            for row_offset, column_offset in edges:
                neighbor_node = (x + row_offset, y + column_offset)
                if neighbor_node in legal_positions:
                    Visualize.__add_edge_to_graph(
                        Visualize.__graph, (x, y), neighbor_node)

    @staticmethod
    def draw_board(state, winner: int, player_1: str, player_2: str, positions=None) -> None:
        board = state[1:]

        # List of all node positions currently filled
        filled_player_1, filled_player_2 = Visualize.__get_filled_nodes(board)
        empty_nodes = Visualize.__get_empty_nodes(board)
        legal_positions = Visualize.__get_legal_positions(board)
        size = Visualize.__board_size
        positions = {}

        # Position nodes to shape a Diamond
        for node in legal_positions:
            positions[node] = (node[0] - node[1], 2 * size - node[1] - node[0])

        plt.axis('off')
        nx.draw_networkx_nodes(Visualize.__graph, pos=positions, nodelist=empty_nodes, node_color='white')
        nx.draw_networkx_edges(Visualize.__graph, pos=positions, alpha=0.5, width=1, edge_color='grey')
        if winner == 1:
            nx.draw_networkx_nodes(Visualize.__graph, pos=positions, nodelist=filled_player_1, node_color='black', label=player_1)
            nx.draw_networkx_nodes(Visualize.__graph, pos=positions, alpha=0.5, nodelist=filled_player_2, node_color='grey', label=player_2)
            plt.legend(prop={'size': 12})
            plt.draw()
            plt.pause(Visualize.__frame_delay * 5)

        elif winner == 2:
            nx.draw_networkx_nodes(Visualize.__graph, pos=positions, alpha=0.5, nodelist=filled_player_1, node_color='grey', label=player_1)
            nx.draw_networkx_nodes(Visualize.__graph, pos=positions, nodelist=filled_player_2, node_color='red', label=player_2)
            plt.legend(prop={'size': 12})
            plt.draw()
            plt.pause(Visualize.__frame_delay * 5)

        else:
            nx.draw_networkx_nodes(Visualize.__graph, pos=positions, nodelist=filled_player_1, node_color='black', label=player_1)
            nx.draw_networkx_nodes(Visualize.__graph, pos=positions, nodelist=filled_player_2, node_color='red', label=player_2)
            plt.legend(prop={'size': 12})
            plt.draw()
            plt.pause(Visualize.__frame_delay)

        plt.clf()

    @staticmethod
    def plot_loss(loss_history: List[int]):
        plt.title('Training loss')
        plt.ylabel('Loss')
        plt.xlabel('Episode')
        plt.plot(loss_history, label="(training) loss")
        plt.legend()
        plt.savefig('plots/loss.png')
        plt.close()

    @staticmethod
    def plot_epsilon(epsilon_history):
        plt.title('Epsilon')
        plt.xlabel('Time step')
        plt.ylabel('$\epsilon$')

        x = [i for i in range(len(epsilon_history))]
        explore_history = [y for y in epsilon_history if y > 0.5]
        exploit_history = [y for y in epsilon_history if y <= 0.5]

        plt.plot(x[:len(explore_history)], explore_history, label='Explorative', color='tab:cyan')
        plt.plot(x[len(explore_history):], exploit_history, label='Exploitative', color='tab:blue')

        plt.legend()
        plt.savefig('plots/epsilon.png')
        plt.close()

    @staticmethod
    def plot_win_statistics(statistics):
        plt.title('TOPP results')
        plt.xlabel('Agent')
        plt.xticks(rotation=45)
        plt.ylabel('wins')
        agents = statistics.keys()
        wins = statistics.values()
        plt.bar(agents, wins)

        plt.tight_layout()
        plt.savefig('plots/TOPP_results.png')
        plt.close()
