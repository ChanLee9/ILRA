#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import difflib
from collections.abc import Mapping
from typing import Any, Callable, Dict


class InterpretorError(ValueError):
    """
    An error raised when the interpretor cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """

    pass


def evaluate(code: str, tools: Dict[str, Callable], state=None, chat_mode=False):
    """
    Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        code (`str`):
            The code to evaluate.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpretorError`.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` should contain the initial inputs but will be
            updated by this function to contain all variables as they are evaluated.
        chat_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the function is called from `Agent.chat`.
    """
    try:
        expression = ast.parse(code)
    except SyntaxError as e:
        print("The code generated by the agent is not valid.\n", e)
        return
    if state is None:
        state = {}
    result = None
    for idx, node in enumerate(expression.body):
        try:
            line_result = evaluate_ast(node, state, tools)
        except InterpretorError as e:
            msg = f"Evaluation of the code stopped at line {idx} before the end because of the following error"
            if chat_mode:
                msg += (
                    f". Copy paste the following error message and send it back to the agent:\nI get an error: '{e}'"
                )
            else:
                msg += f":\n{e}"
            print(msg)
            break
        if line_result is not None:
            result = line_result

    return result


def evaluate_ast(expression: ast.AST, state: Dict[str, Any], tools: Dict[str, Callable]):
    """
    Evaluate an absract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse trough the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abastract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignements.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpretorError`.
    """
    if isinstance(expression, ast.Assign):
        # Assignement -> we evaluate the assignement which should update the state
        # We return the variable assigned as it may be used to determine the final result.
        return evaluate_assign(expression, state, tools)
    elif isinstance(expression, ast.Call):
        # Function call -> we return the value of the function call
        return evaluate_call(expression, state, tools)
    elif isinstance(expression, ast.Constant):
        # Constant -> just return the value
        return expression.value
    elif isinstance(expression, ast.Dict):
        # Dict -> evaluate all keys and values
        keys = [evaluate_ast(k, state, tools) for k in expression.keys]
        values = [evaluate_ast(v, state, tools) for v in expression.values]
        return dict(zip(keys, values))
    elif isinstance(expression, ast.Expr):
        # Expression -> evaluate the content
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.For):
        # For loop -> execute the loop
        return evaluate_for(expression, state, tools)
    elif isinstance(expression, ast.FormattedValue):
        # Formatted value (part of f-string) -> evaluate the content and return
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.If):
        # If -> execute the right branch
        return evaluate_if(expression, state, tools)
    elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.JoinedStr):
        return "".join([str(evaluate_ast(v, state, tools)) for v in expression.values])
    elif isinstance(expression, ast.List):
        # List -> evaluate all elements
        return [evaluate_ast(elt, state, tools) for elt in expression.elts]
    elif isinstance(expression, ast.Name):
        # Name -> pick up the value in the state
        return evaluate_name(expression, state, tools)
    elif isinstance(expression, ast.Subscript):
        # Subscript -> return the value of the indexing
        return evaluate_subscript(expression, state, tools)
    else:
        # For now we refuse anything else. Let's add things as we need them.
        raise InterpretorError(f"{expression.__class__.__name__} is not supported.")


def evaluate_assign(assign, state, tools):
    var_names = assign.targets
    result = evaluate_ast(assign.value, state, tools)

    if len(var_names) == 1:
        state[var_names[0].id] = result
    else:
        if len(result) != len(var_names):
            raise InterpretorError(f"Expected {len(var_names)} values but got {len(result)}.")
        for var_name, r in zip(var_names, result):
            state[var_name.id] = r
    return result


def evaluate_call(call, state, tools):
    if not isinstance(call.func, ast.Name):
        raise InterpretorError(
            f"It is not permitted to evaluate other functions than the provided tools (tried to execute {call.func} of "
            f"type {type(call.func)}."
        )
    func_name = call.func.id
    if func_name not in tools:
        raise InterpretorError(
            f"It is not permitted to evaluate other functions than the provided tools (tried to execute {call.func.id})."
        )

    func = tools[func_name]
    # Todo deal with args
    args = [evaluate_ast(arg, state, tools) for arg in call.args]
    kwargs = {keyword.arg: evaluate_ast(keyword.value, state, tools) for keyword in call.keywords}
    return func(*args, **kwargs)


def evaluate_subscript(subscript, state, tools):
    index = evaluate_ast(subscript.slice, state, tools)
    value = evaluate_ast(subscript.value, state, tools)
    if isinstance(value, (list, tuple)):
        return value[int(index)]
    if index in value:
        return value[index]
    if isinstance(index, str) and isinstance(value, Mapping):
        close_matches = difflib.get_close_matches(index, list(value.keys()))
        if len(close_matches) > 0:
            return value[close_matches[0]]

    raise InterpretorError(f"Could not index {value} with '{index}'.")


def evaluate_name(name, state, tools):
    if name.id in state:
        return state[name.id]
    close_matches = difflib.get_close_matches(name.id, list(state.keys()))
    if len(close_matches) > 0:
        return state[close_matches[0]]
    raise InterpretorError(f"The variable `{name.id}` is not defined.")


def evaluate_condition(condition, state, tools):
    if len(condition.ops) > 1:
        raise InterpretorError("Cannot evaluate conditions with multiple operators")

    left = evaluate_ast(condition.left, state, tools)
    comparator = condition.ops[0]
    right = evaluate_ast(condition.comparators[0], state, tools)

    if isinstance(comparator, ast.Eq):
        return left == right
    elif isinstance(comparator, ast.NotEq):
        return left != right
    elif isinstance(comparator, ast.Lt):
        return left < right
    elif isinstance(comparator, ast.LtE):
        return left <= right
    elif isinstance(comparator, ast.Gt):
        return left > right
    elif isinstance(comparator, ast.GtE):
        return left >= right
    elif isinstance(comparator, ast.Is):
        return left is right
    elif isinstance(comparator, ast.IsNot):
        return left is not right
    elif isinstance(comparator, ast.In):
        return left in right
    elif isinstance(comparator, ast.NotIn):
        return left not in right
    else:
        raise InterpretorError(f"Operator not supported: {comparator}")


def evaluate_if(if_statement, state, tools):
    result = None
    if evaluate_condition(if_statement.test, state, tools):
        for line in if_statement.body:
            line_result = evaluate_ast(line, state, tools)
            if line_result is not None:
                result = line_result
    else:
        for line in if_statement.orelse:
            line_result = evaluate_ast(line, state, tools)
            if line_result is not None:
                result = line_result
    return result


def evaluate_for(for_loop, state, tools):
    result = None
    iterator = evaluate_ast(for_loop.iter, state, tools)
    for counter in iterator:
        state[for_loop.target.id] = counter
        for expression in for_loop.body:
            line_result = evaluate_ast(expression, state, tools)
            if line_result is not None:
                result = line_result
    return result
