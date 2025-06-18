
'''
Data format in align-anything
{
    "question": "...",
    "response_1": "...",
    "response_2": "...",
    "overall_response": "...",
}
Goal key-value format
{  
    "better_text": "...",  
    "worse_text": "...",  
}
'''


@register_template('HOMEWORK')
class HOMEWORK(BaseFormatter):
    system_prompt: str = ''

    def format_preference_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['question']
        better_id = raw_sample['overall_response']
        if better_id not in {'1', '2'}:
            raise ValueError(f"Invalid overall_response id: {better_id}")

        better_response = raw_sample[f"response_{better_id}"]
        worse_response = raw_sample[f"response_{1 if better_id == '2' else 2}"]

        better_conversation = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': better_response},
        ]
        worse_conversation = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': worse_response},
        ]

        meta_info = {
            'better_response': better_response,
            'worse_response': worse_response,
        }

        return better_conversation, worse_conversation, meta_info