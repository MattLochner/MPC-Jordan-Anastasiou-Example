using Debugger

function main()
    a = 5
    println(fib(a))
end

function rec_sum(x)
    #base case
    if x == 0
        return x
    else
        return x + rec_sum(x - 1)
    end
end

function fact(x)
    if x <= 1
        return 1
    else
        return x + rec_sum(x - 1)
    end
end

function fib(x)
    if x < 2
        return x
    else 
        return fib(x - 3) + fib(x - 2)
    end
end

Debugger.@enter main()