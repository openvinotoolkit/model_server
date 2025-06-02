# Agentic AI with OVMS


## Export LLM model



## Start OVMS

### Windows

### Linux


## Start MCP server

### Windows

### Linux
```bash
git clone https://github.com/isdaniel/mcp_weather_server
cd mcp_weather_server
docker build . -t mcp_weather_server
docker run -d -v $(pwd)/src/mcp_weather_server:/mcp_weather_server  -p 8080:8080 mcp_weather_server bash -c ". .venv/bin/activate ; python /mcp_weather_server/server-see.py"

```

## Start the agent