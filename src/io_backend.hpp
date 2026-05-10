#pragma once
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <string>
#include <unistd.h>

class IOBackend {
    int fd;
    std::string path;
    std::mutex io_mutex;

public:
    IOBackend(const std::string& filepath)
        : fd(-1),
          path(filepath) {

        int flags = O_RDWR | O_CREAT | O_DIRECT;
        fd = open(filepath.c_str(), flags, 0644);
        if (fd < 0) {

            flags = O_RDWR | O_CREAT;
            fd = open(filepath.c_str(), flags, 0644);
        }
        if (fd < 0) {
            perror("open");
            exit(1);
        }

        std::cout << "[IOBackend] Opened " << filepath << " (fd: " << fd
                  << ((flags & O_DIRECT) ? ", O_DIRECT" : ", buffered") << ")" << std::endl;
    }

    IOBackend(const IOBackend&) = delete;
    IOBackend& operator=(const IOBackend&) = delete;

    ~IOBackend() {
        if (fd >= 0) {
            std::cout << "[IOBackend] Closing fd: " << fd << std::endl;
            close(fd);
        }
    }

    void write(size_t offset, void* buf, size_t size) {
        std::lock_guard<std::mutex> lock(io_mutex);
        ssize_t ret = pwrite(fd, buf, size, (off_t)offset);
        if (ret != (ssize_t)size) {
            perror("pwrite");
            exit(1);
        }
    }

    void sync() {
        if (fd >= 0) {
            fsync(fd);
        }
    }

    void read(size_t offset, void* buf, size_t size) {
        std::lock_guard<std::mutex> lock(io_mutex);
        ssize_t ret = pread(fd, buf, size, (off_t)offset);
        if (ret != (ssize_t)size) {
            perror("pread");
            exit(1);
        }
    }
};
