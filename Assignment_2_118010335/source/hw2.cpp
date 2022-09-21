#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <sys/time.h>

#define ROW 10
#define COLUMN 50
#define THREAD 10

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 

char map[ROW + 10][COLUMN];

//The current character occupied by the frog
char frog_occupied;

//Suspension parameters
useconds_t suspension;
int max_suspension = 700000;
int min_suspension = 300000;
//The suspension factor sf
//The greater the sf is,
//The longer the suspension is.
float sf = 1;
float max_sf = 5;
float min_sf = 0.05;

//Log parameters
int min_length = 8;
int max_length = 15;

//mutex and condition variable
pthread_mutex_t mutex;
pthread_cond_t cv;

//thread ID array
int thread_ids[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

//The count of logs that have been moved in the current round
int log_moved = 0;

//Indicate the current game status
//0: normal
//1: win
//2: lose
//3: quit
int game_status = 0;

//Timer
struct timeval start_time;
struct timeval end_time;

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

//Generate a random integer between min and max
int random_int(int min, int max) {
	return (int) (rand() % (max - min) + min);
}

//Find frog's y coordinate
//Since the frog moves with the logs horizontally,
//its x coordinate does not change
int find_frog_y() {
	for (int j = 0; j <= COLUMN - 1; j++) {
		if (map[frog.x][j] == '0') {
			return j;
		}
	}
}

//Check if the desired movement is valid
bool check_movable(int dx, int dy) {
	int x = frog.x + dx;
	int y = frog.y + dy;
	if (y < 0 || y > COLUMN - 2) return false;
	else if (x < 0 || x > ROW) return false;
	return true;
}

//Check the current game status
int check_status() {
	if (frog.x == 0)
		return 1;
	else if (frog_occupied == '=' || frog_occupied == '|')
		return 0;
	else if (frog_occupied == ' ')
		return 2;
}

//Move the frog
bool frog_move(int dx, int dy) {
	//Restore the occupied character
	map[frog.x][frog.y] = frog_occupied;
	//Calculate the new coordinates
	frog.x += dx;
	frog.y += dy;
	//Store the occupied character
	frog_occupied = map[frog.x][frog.y];
	//Place the frog
	map[frog.x][frog.y] = '0';
}

//Handle input events
void input_event(char key) {
	//Quit
	if (key == 'q' || key == 'Q') {
		game_status = 3;
	}
	//Adjust speed
	//Speed up
	else if (key == '.') {
		sf /= 2;
		if (sf < min_sf)
			sf = min_sf;
	}
	//Speed down
	else if (key == ',') {
		sf *= 2;
		if (sf > max_sf)
			sf = max_sf;
	}
	//Move
	else if (key == 'w' || key == 'W') {
		if (check_movable(-1, 0)) {
			frog_move(-1, 0);
		}
	}
	else if (key == 's' || key == 'S') {
		if (check_movable(1, 0)) {
			frog_move(1, 0);
		}
	}
	else if (key == 'a' || key == 'A') {
		if (check_movable(0, -1)) {
			frog_move(0, -1);
		}
	}
	else if (key == 'd' || key == 'D') {
		if (check_movable(0, 1)) {
			frog_move(0, 1);
		}
	}
}

//Move the logs & Handle input events & Manage the game
void *logs_move( void *t ){

	/*  Check game's status  */
	/*  Move the logs  */
	int* thread_id = (int*)t;
	while (game_status == 0) {
		//The second to the last threads move the logs
		if (*thread_id != 0) {
			//Suspend the current thread for suspension
			usleep(suspension);
			//After suspension, lock the mutex.
			pthread_mutex_lock(&mutex);
			bool move_right = log_moved % 2 == 0; //Determine the moving direction
			int i = log_moved + 1; //The row index
			int reference; //The index of the column to refer to
			if (move_right) {
				//If the log reaches the left side of the river but the frog still on it, loss
				//map[i][COLUMN - 2] == '=' && map[i][0] == ' ' && frog.x == i
				if (map[i][COLUMN - 2] == '0') {
					game_status = 2;
				}
				char temp = map[i][COLUMN - 2];
				for (int j = COLUMN - 2; j >= 1; j--) { //The column index
					//Calculate the index of the column to refer to
					reference = j - 1;
					map[i][j] = map[i][reference];
				}
				map[i][0] = temp;
			}
			else { //move_left
				//If the log reaches the right side of the river but the frog still on it, loss
				//map[i][0] == '=' && map[i][COLUMN - 2] == ' ' && frog.x == i
				if (map[i][0] == '0') {
					game_status = 2;
				}
				char temp = map[i][0];
				for (int j = 0; j <= COLUMN - 3; j++) { //The column index
					//Calculate the index of the column to refer to
					reference = j + 1;
					map[i][j] = map[i][reference];
				}
				map[i][COLUMN - 2] = temp;
			}
			//If (ROW-1) logs have been moved, signal cv to invoke the first thread
			if (++log_moved == ROW - 1)
				pthread_cond_signal(&cv);
			//Job done, unlock the mutex
			pthread_mutex_unlock(&mutex);
		}

		//The first thread handles input and frog movement
		else if (*thread_id == 0) {

			//Lock the mutex
			pthread_mutex_lock(&mutex);
			//While not all the logs have been moved, release the mutex and wait for signal.
			while (log_moved != ROW - 1)
				pthread_cond_wait(&cv, &mutex);

			//When the logs have been moved (signal received)
			//Update the frog's y coordinate
			//Since the frog moves with the logs horizontally,
			//its x coordinate does not change
			frog.y = find_frog_y();

			//Generate a new random suspension time
			suspension = (int) random_int(min_suspension, max_suspension) * sf;

			//Reset log_moved
			log_moved = 0;

			/*  Check keyboard hits, to change frog's position or quit the game. */
			//Handle the input
			if (kbhit()) {
				char key = getchar();
				//Eliminate special characters
				if (key == 27) {
					while (kbhit())
						getchar();
					key = ' ';
				}
				input_event(key);
			}

			//Update the game status
			if (game_status == 0)
				game_status = check_status();

	/*  Print the map on the screen  */
			printf("\033[0;0H\033[2J");
			for (int i = 0; i <= ROW; i++)
				puts(map[i]);
			printf("Suspension = %d | Suspension factor = %f\n", suspension, sf);
			printf("Press , and . to adjust speed\n");

			gettimeofday(&end_time, NULL);
			float duration = (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec);
			printf("Your time = %f seconds\n", duration / 1000000);

			//Job done, unlock the mutex
			pthread_mutex_unlock(&mutex);
		}
	}

	pthread_exit(NULL);
}

//Initialize the floating logs randomly
void init_logs() {
	for (int i = 1; i < ROW; i++) {
		//Generate a random length
		int length = (rand() % (max_length - min_length)) + min_length;
		//Generate a random position (left index)
		int left_index = rand() % ((COLUMN - 1) - 1 - (length - 1) + 1);
		//Place the log
		int j;
		for (j = 0; j < left_index; j++) {
			map[i][j] = ' ';
		}
		for (int j = left_index; j < left_index + length; j++) {
			map[i][j] = '=';
		}
		for (int j = left_index + length; j < COLUMN - 1; j++) {
			map[i][j] = ' ';
		}
	}
}

//Output the game result
void output_result() {
	printf("\033[14;0H\033[K");
	printf("\033[13;0H\033[K");
	printf("\033[12;0H\033[K");
	switch (game_status)
	{
	case 1:
		printf("You Win\n");
		break;
	case 2:
		printf("You Lose\n");
		break;
	case 3:
		printf("You Quit\n");
		break;
	default:
		printf("Error\n");
		break;
	}
	gettimeofday(&end_time, NULL);
	float duration = (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec);
	printf("Your time = %f seconds\n", duration / 1000000);

}

int main( int argc, char *argv[] ){

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 

	//Remember the occupied character
	frog_occupied = map[frog.x][frog.y];

	map[frog.x][frog.y] = '0' ; 

	init_logs();

	//Print the map into screen
	for (i = 0; i <= ROW; ++i)
		puts(map[i]);

	//Generate the initial random suspension time
	srand((unsigned)time(0));
	suspension = (int)random_int(min_suspension, max_suspension) * sf;

	//Start timing
	gettimeofday(&start_time, NULL);

	/*  Create pthreads for wood move and frog control.  */

	//Declare the threads
	pthread_t threads[10];

	//Initialize the attribute
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	//Initialize the mutex and condition variable
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cv, NULL);

	//Create threads
	for (int i = 0; i < THREAD; i++)
		pthread_create(&threads[i], &attr, logs_move, (void*)&thread_ids[i]);

	//Join the threads
	for (int i = 0; i < THREAD; i++)
		pthread_join(threads[i], NULL);

	/*  Display the output for user: win, lose or quit.  */
	output_result();

	//Destroy the pthread stuff
	pthread_attr_destroy(&attr);
	pthread_mutex_destroy(&mutex);
	pthread_cond_destroy(&cv);

	//Exit
	pthread_exit(NULL);

	return 0;

}
