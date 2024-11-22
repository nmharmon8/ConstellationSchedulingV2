import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
import ollama
from tqdm import tqdm
import time

example_imaging_task = {"name":"TARGET", "task_type":"1", "lat": "LATITUDE", "lon":"LONGITUDE", "priority":"1", "duration":"15", "min_elev":"0.785"}
#example_imaging_task = {"target":"TARGET", "collect_type":"EO", "reasoning":"REASONING", "latitude": "LATITUDE", "longitude":"LONGITUDE", "quality":"QUALITY", "uuid":"UUID"}
example_rf_task = {"target":"TARGET", "collect_type":"RF", "reasoning":"REASONING", "center_frequency_mhz":"CENTER_FREQUENCY_MHZ", "sample_rate_msps":"SAMPLE_RATE_MSPS", "latitude": "LATITUDE", "longitude":"LONGITUDE", "uuid":"UUID"}
example_sar_task = {"target":"TARGET", "collect_type":"SAR", "reasoning":"REASONING", "latitude": "LATITUDE", "longitude":"LONGITUDE", "mode":"MODE", "uuid":"UUID"}
example_ir_task = {"target":"TARGET", "collect_type":"IR", "reasoning":"REASONING", "latitude": "LATITUDE", "longitude":"LONGITUDE", "ref_uuid":"UUID"}


api_queries = [
    f"with a single json message following exactly this Image Task Format (json list): [{example_imaging_task}]. Fill in lat lon and name for only relevant locations for the prompt",
    f"with a single json message following exactly this RF Task Format (json list): [{example_rf_task}].Provide a concise reasoning of how it is relevant to the promp with no commas. Choose multiples of 10MSps for sample rate up to a maximum of 100MHz. Choose a center frequency in the middle if there are multiple frequencies of interest.",
    f"with a single json message following exactly this SAR Task Format (json list): [{example_sar_task}]. Provide a concise reasoning of how it is relevant to the prompt with no commas. Possible modes are: spot, site, strip, and scan. Do not change collect_type. ",
    f"with a single json message following exactly this Infrared Task Format (json list): [{example_ir_task}]. Be sure to collect all relevant areas. Provide a concise reasoning of how it is relevant to the prompt with no commas. Do not change collect_type. This modality is good for fire monitoring."
]

print(api_queries)

documents = api_queries

while True:
    try:
        client = weaviate.connect_to_local()
        client.is_live()  # Check if Weaviate is live
        print("Connected to Weaviate successfully.")
        break
    except Exception as e:
        print("Waiting for Weaviate to be available...")
        time.sleep(5)  # Wait before retrying


#Only do this once
client = weaviate.connect_to_local()
# Create a new data collection


collection_name = "apis"

try:
    client.collections.delete(collection_name)
except:
    print("Delete didnt work (probably not an issue)")

collection = client.collections.create(
    name = collection_name, # Name of the data collection
    properties=[
        Property(name="text", data_type=DataType.TEXT), # Name and data type of the property
    ],
)

with collection.batch.dynamic() as batch:
    for i, d in tqdm(enumerate(documents), total=len(documents)):
        response = ollama.embeddings(model = "all-minilm",
                                    prompt = d)
        # Add data object with text and embedding
        batch.add_object(
            properties = {"text" : d},
            vector = response["embedding"],
        )

client.close()

print("done!")
