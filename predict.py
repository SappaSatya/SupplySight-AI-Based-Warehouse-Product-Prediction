from mcp.server.fastmcp import FastMCP
from server.predict_impl import predict_product_weight

mcp = FastMCP("predict_demand")

USER_AGENT = "predict_demand-app/1.0"

@mcp.tool()
async def get_product_weight() -> str:
    """Predict the product weight in tons
    
    Args:
    
    """
    try:
        weight = predict_product_weight()
        return f"Predicted weight in tons: {weight}"
    except Exception as e:
        return f"Error predicting weight: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
