#import <Foundation/Foundation.h>

NSString *NMOEChatSystemPrompt(void) {
    return @"<|im_start|>system\n"
    @"You are a helpful local assistant running inside nmoe. Answer clearly and concisely.\n"
    @"<|im_end|>\n";
}

NSString *NMOEChatUserPrompt(NSString *userText) {
    return [NSString stringWithFormat:@"<|im_start|>user\n%@<|im_end|>\n<|im_start|>assistant\n", userText ?: @""];
}
