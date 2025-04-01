from pydantic import BaseModel


class BenchmarkResults(BaseModel):
    N_Clients: int
    N_Inputs: int
    Time_Preprocessing: float
    Sent_Preprocessing: int
    Time_Local_Commitment: float
    Sent_Local_Commitment: int
    Time_Cosine_Similarity: float
    Sent_Cosine_Similarity: int
    Time_Euclidean_Norm: float
    Sent_Euclidean_Norm: int
    Time_Meta_Clipping: float
    Sent_Meta_Clipping: int
    Time_Aggregation: float
    Sent_Aggregation: int
    Time_Total: float
    Sent_Total: int
