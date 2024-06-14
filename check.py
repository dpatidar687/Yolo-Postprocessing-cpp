import build.run_yolo_onnx
import ctypes

# Create an instance of the Check class
obj = build.run_yolo_onnx.Check("test")

# Call the getString method to get the string
string_address = obj.getString()

# Print the string address
print("String address:", string_address)

# Access the string value using its memory address
# string_value = ctypes.cast(int(string_address, 16), ctypes.c_char_p).value.decode("utf-8")

# # Print the string value
# print("String value:", string_value)
