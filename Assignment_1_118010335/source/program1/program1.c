#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){

	//detect invalid input
	if (argc < 2) {
		printf("Invalid input\n");
		exit(1);
	}

	/* fork a child process */
	int status;
	pid_t pid = fork();

	//fork error
	if (pid < 0) {
		printf("Fork error!\n");
	}

	//normal fork
	else {

	/* execute test program */

		//Child process
		if (pid == 0) {
			printf("This is the child process.\n");
			printf("Child process id is %d\n", getpid());
			printf("Child process start to execute test program:\n");

			//modify argv
			int i;
			char* arg[argc];
			for (i = 0; i < argc - 1; i++) {
				arg[i] = argv[i + 1];
			}
			arg[argc - 1] = NULL;
			execve(arg[0], arg, NULL);

			//go back to original child process -> error
			printf("Continue to run original child process!\n");
			perror("execve");
			exit(EXIT_FAILURE);
		}

	/* wait for child process terminates */

		//Parent process
		else {
			printf("This is the parent process.\n");
			printf("Parent process id is %d\n", getpid());

			waitpid(-1, &status, WUNTRACED);

			printf("Parent process receives the SIGCHLD signal\n");

	/* check child process'  termination status */
			
			//normal
			if (WIFEXITED(status)) {
				printf("Normal termination with EXIT STATUS = %d\n", WEXITSTATUS(status));
			}
			
			//signaled
			else if (WIFSIGNALED(status)) {
				char* signal = NULL;
				char* description = NULL;
				switch (WTERMSIG(status)) {
				case SIGHUP:
					signal = "SIGHUP";
					description = "hang up";
					break;
				case SIGINT:
					signal = "SIGINT";
					description = "interrupt";
					break;
				case SIGQUIT:
					signal = "SIGQUIT";
					description = "quit";
					break;
				case SIGILL:
					signal = "SIGILL";
					description = "illegal";
					break;
				case SIGTRAP:
					signal = "SIGTRAP";
					description = "trap";
					break;
				case SIGABRT:
					signal = "SIGABRT";
					description = "abort";
					break;
				case SIGBUS:
					signal = "SIGBUS";
					description = "bus";
					break;
				case SIGFPE:
					signal = "SIGFPE";
					description = "floating point exception";
					break;
				case SIGKILL:
					signal = "SIGKILL";
					description = "kill";
					break;
				case SIGSEGV:
					signal = "SIGSEGV";
					description = "segmentation fault";
					break;
				case SIGPIPE:
					signal = "SIGPIPE";
					description = "pipe";
					break;
				case SIGALRM:
					signal = "SIGALRM";
					description = "alarm";
					break;
				case SIGTERM:
					signal = "SIGTERM";
					description = "terminate";
					break;
				default:
					break;
				}

				if (signal) {
					printf("child process get %s signal\n", signal);
					printf("child process is abort by %s signal\n", description);
					printf("CHILD EXECUTION FAILED!!\n");
				}
				else {
					printf("\n!!UNKNOWN SIGNAL!!\n");
				}

			}

			//stopped
			else if (WIFSTOPPED(status)) {
				if (WSTOPSIG(status) == SIGSTOP) {
					printf("child process get SIGSTOP signal\n");
					printf("child process stopped\n");
					printf("CHILD EXECUTION STOPPED\n");
				}
				else {
					printf("\n!!UNKNOWN SIGNAL!!\n");
				}
			}

			//continued
			else {
				printf("CHILD PROCESS CONTINUED\n");
			}

			exit(0);
		}
	}
}
