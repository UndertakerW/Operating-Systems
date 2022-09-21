#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

//recursively fork the process
void my_fork(char* arg[], int argc, int index, pid_t* pids, int* signals) {

	int status;

	//vfork(): let the processes share the heap
	pid_t pid = vfork();

	//fork error
	if (pid == -1) {
		perror("fork");
		exit(1);
	}
	else {
		//child process
		if (pid == 0) {
			//the last child: execve() directly
			if (index == argc - 2)
				execve(arg[argc - 2], arg, NULL);
			//the children in between: my_fork()
			else
				my_fork(arg, argc, index + 1, pids, signals);
		}
		//parent process
		else {
			//wait for child process
			waitpid(pid, &status, WUNTRACED);

			//update the process tree
			pids[index] = pid;
			signals[index] = status;

			//the parents: execve() after children
			if (index > 0)
				execve(arg[index - 1], arg, NULL);
		}
	}
}

//print the process tree
void print_process_tree(int argc, pid_t* pids) {
	printf("The process tree: ");
	printf("%d", getpid());
	for (int i = 0; i < argc - 1; i++)
		printf("->%d", pids[i]);
	printf("\n");
}

//print the process information
void print_info(int argc, pid_t* pids, int* signals) {
	int child;
	int ppid;

	for (int i = 0; i < argc - 1; i++) {
		//the children
		if (i < argc - 2) {
			child = argc - 2 - i;
			ppid = pids[argc - 3 - i];
		}
		//the first process
		else {
			child = 0;
			ppid = getpid();
		}

		//normal
		if (WIFEXITED(signals[child])) {
			printf("The child process (pid=%d) of parent process (pid=%d) has normal execution\n", pids[child], ppid);
			printf("Its exit status = 0\n\n");
		}

		//signaled
		else if (WIFSIGNALED(signals[child])) {
			char* signal = NULL;
			char* description = NULL;
			switch (WTERMSIG(signals[child])) {
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
				printf("The child process (pid=%d) of parent process (pid=%d) is stopped by signal\n", pids[child], ppid);
				printf("Its signal number = %d\n", WTERMSIG(signals[child]));
				printf("Child process get %s signal\n", signal);
				printf("Child was terminated by %s signal\n\n", description);
			}
			else {
				printf("\n!!UNKNOWN SIGNAL!!\n");
			}

		}

		//stopped
		else if (WIFSTOPPED(signals[child])) {
			if (WSTOPSIG(signals[child]) == SIGSTOP) {
				printf("The child process (pid=%d) of parent process (pid=%d) is stopped by signal\n", pids[child], ppid);
				printf("Its signal number = %d\n", WSTOPSIG(signals[child]));
				printf("Child process get SIGSTOP signal\n");
				printf("Child process is stopped\n\n");;
			}
			else {
				printf("\n!!UNKNOWN SIGNAL!!\n");
			}
		}

		//continued
		else {
			printf("CHILD PROCESS CONTINUED\n");
		}
	}
}

int main(int argc, char *argv[]){

	/* Implement the functions here */

	//detect invalid input
	if (argc < 2) {
		printf("Invalid input\n");
		exit(1);
	}

	//the process tree
	pid_t* pids = calloc(256, sizeof(int));
	int* signals = calloc(256, sizeof(int));

	//modify argv[]
	int i;
	char* arg[argc];
	for (i = 0; i < argc - 1; i++) {
		arg[i] = argv[i + 1];
	}
	arg[argc - 1] = NULL;

	//fork and execute
	my_fork(arg, argc, 0, pids, signals);

	//print the process tree
	print_process_tree(argc, pids);

	//print the process information
	print_info(argc, pids, signals);

	//free heap memory
	free(pids);
	free(signals);

	printf("Myfork process (pid=%d) execute normally\n", getpid());

	return 0;
}
