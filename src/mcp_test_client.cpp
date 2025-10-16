#include <iostream>
#include <string>
#include <mcp_sse_client.h>
#include <mcp_message.h>
#include <mcp_logger.h>

int main() {
    try {
        // Initialize logger
        mcp::logger::instance().set_level(mcp::log_level::info);
        
        std::cout << "MCP SSE Client Test Program" << std::endl;
        std::cout << "MCP Version: " << mcp::MCP_VERSION << std::endl;

        // Create an SSE client to connect to http://localhost:8000/sse
        mcp::sse_client client("http://localhost:8000", "/sse");
        
        std::cout << "Created MCP SSE client for http://localhost:8000/sse" << std::endl;
        
        // Set capabilites
        mcp::json capabilities = {
            {"roots", {{"listChanged", true}}}
        };
        client.set_capabilities(capabilities);
        // Set timeout
        client.set_timeout(10);

        // Try to initialize the client
        bool initialized = client.initialize("ovms-test-client", mcp::MCP_VERSION);
        
        if (initialized) {
            std::cout << "Client initialized successfully" << std::endl;
            
            // Try to ping the server
            bool ping_result = client.ping();
            std::cout << "Ping result: " << (ping_result ? "SUCCESS" : "FAILED") << std::endl;
            
            if (ping_result) {
                // Get available tools
                try {
                    std::vector<mcp::tool> tools = client.get_tools();
                    std::cout << "Available tools: " << tools.size() << std::endl;
                    
                    // Look for Python execution tool
                    bool python_tool_found = false;
                    for (const auto& tool : tools) {
                        std::cout << "Tool: " << tool.name << " - " << tool.description << std::endl;
                        if (tool.name == "python" || tool.name == "execute_python" || tool.name == "python_execute") {
                            python_tool_found = true;
                            
                            // Execute Python code: print('hello')
                            std::cout << "Executing Python code: print('hello')" << std::endl;
                            mcp::json args = mcp::json::object();
                            args["code"] = "print('hello')";
                            
                            mcp::json result = client.call_tool(tool.name, args);
                            std::cout << "Python execution result: " << result.dump(2) << std::endl;
                            break;
                        }
                    }
                    
                    if (!python_tool_found) {
                        std::cout << "Python execution tool not found. Available tools:" << std::endl;
                        for (const auto& tool : tools) {
                            std::cout << "  - " << tool.name << std::endl;
                        }
                        
                        // Try to call a generic tool if any exist
                        if (!tools.empty()) {
                            std::cout << "Trying to call first available tool: " << tools[0].name << std::endl;
                            mcp::json args = mcp::json::object();
                            if (tools[0].name.find("python") != std::string::npos) {
                                args["code"] = "print('hello')";
                            }
                            mcp::json result = client.call_tool(tools[0].name, args);
                            std::cout << "Tool execution result: " << result.dump(2) << std::endl;
                        }
                    }
                } catch (const mcp::mcp_exception& e) {
                    std::cerr << "Error getting tools or executing Python: " << e.what() << std::endl;
                }
            }
        } else {
            std::cout << "Client initialization failed" << std::endl;
        }

    } catch (const mcp::mcp_exception& e) {
        std::cerr << "MCP Exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "MCP SSE Client test completed" << std::endl;
    return 0;
}