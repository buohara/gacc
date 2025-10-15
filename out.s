section .data
fmt: db "%d", 10, 0
a: dq 0
b: dq 0
c: dq 0

section .text
global main
extern printf

main:
    mov qword [a], 1000
    mov qword [b], 3000
    mov rax, [a]
    add rax, [b]
    mov [c], rax
    sub rsp, 8
    mov rdi, fmt
    mov rsi, [c]
    xor rax, rax
    call printf
    add rsp, 8
    mov eax, 0
    ret